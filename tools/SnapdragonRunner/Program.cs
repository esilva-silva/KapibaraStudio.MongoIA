using System.Diagnostics;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Microsoft.ML.OnnxRuntimeGenAI;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2) {
    if (i + 1 >= args.Length || !args[i].StartsWith("--")) throw new ArgumentException("Use --model PATH --corpus PATH --out PATH --backend dml");
    options[args[i][2..]] = args[i + 1];
}
if (options.GetValueOrDefault("backend") == "dml-smoke") return GpuProbe.Run(options["model"], options["out"]);
var rows = new List<object>();
var report = new Dictionary<string, object?> {
    ["schema"] = "kapi-snapdragon-sample-v1", ["context_tokens"] = 2048, ["max_new_tokens"] = 32,
    ["greedy"] = true, ["items"] = rows,
    ["backend"] = options.GetValueOrDefault("backend", "dml"), ["model"] = options.GetValueOrDefault("model"),
    ["provider_evidence"] = new { requested = "DmlExecutionProvider", configured = "dml", effective = (string?)null,
        verified = false, reason = "Explicit DirectML configuration; per-node profiling not exposed by this harness. GPU execution is unverified." }
};
int exitCode = 0;
string profilePrefix = Path.GetFullPath(options["out"]) + ".profile-" + Guid.NewGuid().ToString("N");
Directory.CreateDirectory(Path.GetDirectoryName(profilePrefix)!);
try {
    var corpusLines = File.ReadLines(options["corpus"]).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();
    if (corpusLines.Length == 0) throw new ArgumentException("Corpus is empty");
    using (var corpusStream = File.OpenRead(options["corpus"])) report["corpus_sha256"] = Convert.ToHexString(SHA256.HashData(corpusStream)).ToLowerInvariant();
    if (options.GetValueOrDefault("backend", "dml") != "dml") throw new ArgumentException("This .NET runner currently supports dml only");
    using var config = new Config(options["model"]);
    config.ClearProviders();
    config.AppendProvider("dml");
    config.SetProviderOption("dml", "device_id", options.GetValueOrDefault("device-id", "0"));
    var original = JsonNode.Parse(File.ReadAllText(Path.Combine(options["model"], "genai_config.json")))!;
    var decoder = original["model"]!["decoder"]!;
    var decoderOverlay = new JsonObject { ["session_options"] = new JsonObject { ["enable_profiling"] = profilePrefix } };
    if (decoder["pipeline"] is JsonArray pipeline) {
        var stages = new JsonArray();
        foreach (var stage in pipeline) {
            var copy = stage!.DeepClone();
            copy["session_options"] ??= new JsonObject();
            copy["session_options"]!["enable_profiling"] = profilePrefix + "-" + stages.Count;
            stages.Add(copy);
        }
        decoderOverlay["pipeline"] = stages;
    }
    config.Overlay(new JsonObject { ["model"] = new JsonObject { ["decoder"] = decoderOverlay } }.ToJsonString());
    using var model = new Model(config);
    using var tokenizer = new Tokenizer(model);
    using var modelConfig = JsonDocument.Parse(File.ReadAllText(Path.Combine(options["model"], "genai_config.json")));
    var eos = new HashSet<int>();
    if (modelConfig.RootElement.GetProperty("model").TryGetProperty("eos_token_id", out var eosNode)) {
        if (eosNode.ValueKind == JsonValueKind.Array) foreach (var v in eosNode.EnumerateArray()) eos.Add(v.GetInt32());
        else if (eosNode.ValueKind == JsonValueKind.Number) eos.Add(eosNode.GetInt32());
    }
    var stops = new HashSet<int>(eos);
    foreach (var marker in new[] { "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>" }) {
        using var encoded = tokenizer.Encode(marker);
        if (encoded[0].Length == 1) stops.Add(encoded[0][0]);
    }
    foreach (var line in corpusLines) {
        using var doc = JsonDocument.Parse(line);
        var item = doc.RootElement;
        var row = new Dictionary<string, object?> { ["id"] = item.GetProperty("id").Clone() };
        try {
            var prompt = item.TryGetProperty("prompt_text", out var p) ? p.GetString()! : item.GetProperty("prompt").GetString()!;
            using var input = tokenizer.Encode(prompt);
            int promptLength = input[0].Length;
            if (promptLength == 0 || promptLength + 32 > 2048) throw new ArgumentException("Prompt exceeds context budget or is empty; no implicit truncation");
            if (item.TryGetProperty("prompt_ids", out var expected) && !input[0].ToArray().SequenceEqual(expected.EnumerateArray().Select(v => v.GetInt32()))) throw new ArgumentException("Tokenizer mismatch with corpus prompt_ids");
            using var parameters = new GeneratorParams(model);
            parameters.SetSearchOption("do_sample", false);
            parameters.SetSearchOption("max_length", promptLength + 32);
            using var generator = new Generator(model, parameters);
            var tokens = new List<int>();
            var suffix = item.TryGetProperty("suffix", out var s) ? s.GetString() ?? "" : "";
            var watch = Stopwatch.StartNew();
            double? first = null;
            string stop = "runtime_done";
            generator.AppendTokenSequences(input);
            while (!generator.IsDone() && tokens.Count < 32) {
                int beforeLength = generator.GetSequence(0).Length;
                generator.GenerateNextToken();
                int afterLength = generator.GetSequence(0).Length;
                if (afterLength <= beforeLength) {
                    if (!generator.IsDone()) throw new InvalidOperationException("Generator made no sequence progress");
                    stop = "runtime_done_without_token";
                    break;
                }
                first ??= watch.Elapsed.TotalSeconds;
                int token = generator.GetSequence(0)[^1];
                tokens.Add(token);
                if (stops.Contains(token)) { stop = eos.Contains(token) ? "eos" : "fim_stop"; break; }
                if (suffix.Length > 0 && tokenizer.Decode(tokens.ToArray()).Contains(suffix)) { stop = "suffix"; break; }
                if (tokens.Count == 32) stop = "max_tokens";
            }
            double elapsed = watch.Elapsed.TotalSeconds;
            var content = tokens.Where(t => !stops.Contains(t)).ToArray();
            row["generated_ids"] = tokens;
            var text = tokenizer.Decode(content);
            if (suffix.Length > 0 && text.Contains(suffix)) text = text[..text.IndexOf(suffix, StringComparison.Ordinal)];
            row["generated_text"] = text;
            row["prompt_tokens"] = promptLength;
            row["generated_tokens"] = tokens.Count;
            row["content_tokens"] = content.Length;
            row["ttft_ms"] = first * 1000;
            row["elapsed_s"] = elapsed;
            row["decode_tokens_per_second"] = tokens.Count > 1 && elapsed > first ? (tokens.Count - 1) / (elapsed - first) : null;
            row["stop_reason"] = stop;
            row["eos"] = stop == "eos";
        } catch (Exception ex) { row["error"] = ex.ToString(); exitCode = 1; }
        using var process = Process.GetCurrentProcess();
        row["memory"] = new { rss_bytes = process.WorkingSet64, peak_rss_bytes = process.PeakWorkingSet64 };
        rows.Add(row);
        if (rows.Count % 10 == 0) {
            Console.Error.WriteLine($"Processed {rows.Count}/{corpusLines.Length} requests");
            report["status"] = "running";
            report["processed"] = rows.Count;
            report["total"] = corpusLines.Length;
            File.WriteAllText(Path.GetFullPath(options["out"]) + ".progress.json", JsonSerializer.Serialize(report));
        }
    }
} catch (Exception ex) { report["error"] = ex.ToString(); exitCode = 1; }
var profileFiles = Directory.GetFiles(Path.GetDirectoryName(profilePrefix)!, Path.GetFileName(profilePrefix) + "*.json");
var providerCounts = new Dictionary<string, int>();
var profileErrors = new List<string>();
foreach (var file in profileFiles) {
    try {
        using var profile = JsonDocument.Parse(File.ReadAllText(file));
        foreach (var entry in profile.RootElement.EnumerateArray()) {
            if (entry.TryGetProperty("args", out var info) && info.TryGetProperty("provider", out var provider)) {
                var name = provider.GetString()!;
                if (!string.IsNullOrWhiteSpace(name)) providerCounts[name] = providerCounts.GetValueOrDefault(name) + 1;
            }
        }
    } catch (Exception ex) { profileErrors.Add(ex.Message); }
}
report["provider_evidence"] = new { requested = "DmlExecutionProvider", configured = "dml", effective = providerCounts.Keys.ToArray(),
    verified = providerCounts.GetValueOrDefault("DmlExecutionProvider") > 0, kernel_events_by_provider = providerCounts,
    cpu_kernel_events = providerCounts.GetValueOrDefault("CPUExecutionProvider"), profile_files = profileFiles,
    profile_errors = profileErrors, reason = providerCounts.Count > 0 ? "Observed ORT kernel events after model disposal" : "No provider kernel events captured; GPU execution unverified" };
report["status"] = exitCode == 0 ? "completed" : "failed";
using (var process = Process.GetCurrentProcess()) report["memory_after"] = new { rss_bytes = process.WorkingSet64, peak_rss_bytes = process.PeakWorkingSet64 };
var output = Path.GetFullPath(options["out"]);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
return exitCode;
