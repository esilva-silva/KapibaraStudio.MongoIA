using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>Debug prefill only; output statistics do not qualify generation quality.</summary>
public static class DmlPrefill
{
    public static int Run(Dictionary<string, string> args)
    {
        string outputPath = Path.GetFullPath(args["out"]);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        int kvHeads = int.Parse(args.GetValueOrDefault("kv-heads", "2"));
        int headSize = int.Parse(args.GetValueOrDefault("head-size", "128"));
        var rows = new List<object>();
        var providers = new Dictionary<string, int>();
        var watch = Stopwatch.StartNew();
        string? error = null, profilePath = null;
        InferenceSession? session = null;
        try
        {
            using var item = JsonDocument.Parse(File.ReadLines(args["corpus"]).First(x => !string.IsNullOrWhiteSpace(x)));
            long[] ids = item.RootElement.GetProperty("prompt_ids").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            int length = ids.Length;
            if (length is < 1 or > 2048 || kvHeads < 1 || headSize < 1)
                throw new ArgumentException("Invalid prefill or KV dimensions");
            using var options = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableMemoryPattern = false,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
                ProfileOutputPathPrefix = outputPath + ".profile",
                EnableProfiling = true
            };
            options.AppendExecutionProvider_DML(0);
            Console.Error.WriteLine("Loading DML prefill probe");
            session = new InferenceSession(args["model"], options);
            var inputs = new List<NamedOnnxValue>();
            foreach (var (name, meta) in session.InputMetadata)
            {
                if (name is "input_ids" or "attention_mask" or "position_ids")
                {
                    if (meta.ElementType != typeof(long)) throw new ArgumentException("Expected INT64 " + name);
                    long[] values = name == "input_ids" ? ids : name == "attention_mask"
                        ? Enumerable.Repeat(1L, length).ToArray()
                        : Enumerable.Range(0, length).Select(x => (long)x).ToArray();
                    inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(values, new[] { 1, length })));
                }
                else if (name.StartsWith("past_key_values.", StringComparison.Ordinal) && meta.ElementType == typeof(Float16))
                {
                    inputs.Add(NamedOnnxValue.CreateFromTensor(name,
                        new DenseTensor<Float16>(Array.Empty<Float16>(), new[] { 1, kvHeads, 0, headSize })));
                }
                else throw new ArgumentException("Unsupported input " + name);
            }
            Console.Error.WriteLine("Running DML prefill probe");
            using var results = session.Run(inputs);
            foreach (var output in results)
            {
                IEnumerable<float> values;
                int[] dimensions;
                long count;
                if (session.OutputMetadata[output.Name].ElementType == typeof(float))
                {
                    var tensor = output.AsTensor<float>();
                    values = tensor; dimensions = tensor.Dimensions.ToArray(); count = tensor.Length;
                }
                else if (session.OutputMetadata[output.Name].ElementType == typeof(Float16))
                {
                    var tensor = output.AsTensor<Float16>();
                    values = tensor.Select(v => (float)v); dimensions = tensor.Dimensions.ToArray(); count = tensor.Length;
                }
                else throw new ArgumentException("Unsupported output " + output.Name);
                if (args.TryGetValue("dump-prefix", out var dumpPrefix) &&
                    (output.Name == "/model/embed_tokens/Gather/output_0" ||
                     output.Name == "__rmsnorm_decomposed_1_square" || output.Name == "__rmsnorm_decomposed_1_mean"))
                {
                    string suffix = output.Name.Contains("Gather") ? "embedding" : output.Name.EndsWith("square") ? "square" : "mean";
                    string dumpPath = Path.GetFullPath(dumpPrefix + "." + suffix + ".f32");
                    Directory.CreateDirectory(Path.GetDirectoryName(dumpPath)!);
                    File.WriteAllBytes(dumpPath, MemoryMarshal.AsBytes(values.ToArray().AsSpan()).ToArray());
                }
                var sample = new List<float?>();
                long nonfinite = 0, index = 0;
                double? maximum = null;
                int argmax = -1;
                float best = float.NegativeInfinity;
                foreach (float value in values)
                {
                    bool finite = float.IsFinite(value);
                    if (sample.Count < 16) sample.Add(finite ? value : null);
                    if (!finite) nonfinite++;
                    else
                    {
                        maximum = Math.Max(maximum ?? 0, Math.Abs(value));
                        if (output.Name == "logits" && index >= count - dimensions[^1] && value > best)
                        { best = value; argmax = (int)(index % dimensions[^1]); }
                    }
                    index++;
                }
                rows.Add(new { name = output.Name, shape = dimensions, sample, nonfinite,
                    finite_abs_max = maximum, last_argmax = argmax });
            }
        }
        catch (Exception ex) { error = ex.ToString(); }
        finally
        {
            if (session is not null)
            {
                try { profilePath = session.EndProfiling(); }
                catch (Exception ex) { error ??= ex.ToString(); }
                session.Dispose();
            }
        }
        if (profilePath is not null && File.Exists(profilePath))
        {
            using var profile = JsonDocument.Parse(File.ReadAllText(profilePath));
            foreach (var entry in profile.RootElement.EnumerateArray())
                if (entry.TryGetProperty("args", out var info) && info.TryGetProperty("provider", out var provider))
                {
                    string name = provider.GetString()!;
                    providers[name] = providers.GetValueOrDefault(name) + 1;
                }
        }
        using var binary = File.OpenRead(Assembly.GetExecutingAssembly().Location);
        var report = new { schema = "kapi-dml-prefill-probe-v2", model = args["model"], device_id = 0,
            kv_heads = kvHeads, head_size = headSize, outputs = rows, profile = profilePath,
            kernel_events_by_provider = providers, cpu_fallback_disabled = false,
            runner_binary_sha256 = Convert.ToHexString(SHA256.HashData(binary)).ToLowerInvariant(),
            elapsed_s = watch.Elapsed.TotalSeconds, error,
            scope = "Instrumented single prefill; not generation quality or full model qualification" };
        File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        return error is null ? 0 : 1;
    }
}
