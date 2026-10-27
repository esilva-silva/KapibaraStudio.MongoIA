using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>Proves execution of a small FP32 MatMul/Relu graph using DirectML kernel profiling.</summary>
public static class GpuProbe
{
    public static int Run(string modelPath, string outputPath)
    {
        var observed = new HashSet<string>(StringComparer.Ordinal);
        var kernels = new List<object>();
        float[] input = [-1f, 0f, 1f, 2f];
        float[] expected = [0f, 0f, 1f, 2f];
        float[]? actual = null;
        string? profilePath = null;
        string? error = null;
        bool passed = false;
        bool fallbackDisabled = false;
        InferenceSession? session = null;
        bool profilingEnded = false;
        string absoluteOutput = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(absoluteOutput)!);
        try
        {
            using var options = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableMemoryPattern = false,
                ProfileOutputPathPrefix = Path.Combine(Path.GetDirectoryName(absoluteOutput)!, "gpu-probe-profile"),
                EnableProfiling = true
            };
            options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");
            fallbackDisabled = true;
            options.AppendExecutionProvider_DML(0);
            session = new InferenceSession(modelPath, options);
            if (session.InputMetadata.Count != 1)
                throw new InvalidOperationException("Probe graph must have exactly one input.");
            var inputName = session.InputMetadata.Keys.Single();
            var tensor = new DenseTensor<float>(input, new[] { 1, 4 });
            using (var result = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) }))
            {
                if (result.Count != 1) throw new InvalidOperationException("Probe graph must have exactly one output.");
                var output = result.First().AsTensor<float>();
                if (!output.Dimensions.SequenceEqual(new[] { 1, 4 }))
                    throw new InvalidOperationException("Unexpected probe output dimensions.");
                actual = output.ToArray();
            }
            profilePath = session.EndProfiling();
            profilingEnded = true;
            using var profile = JsonDocument.Parse(File.ReadAllText(profilePath));
            foreach (var item in profile.RootElement.EnumerateArray())
            {
                if (!item.TryGetProperty("name", out var name) ||
                    !name.GetString()!.EndsWith("_kernel_time", StringComparison.Ordinal) ||
                    !item.TryGetProperty("args", out var arguments) ||
                    !arguments.TryGetProperty("provider", out var providerNode)) continue;
                string? provider = providerNode.GetString();
                if (string.IsNullOrWhiteSpace(provider)) continue;
                observed.Add(provider);
                kernels.Add(new { name = name.GetString(), provider });
            }
            bool correct = actual.Length == expected.Length && actual.Zip(expected).All(p => Math.Abs(p.First - p.Second) <= 1e-6f);
            bool gpu = observed.Contains("DmlExecutionProvider");
            bool cpu = observed.Contains("CPUExecutionProvider");
            passed = correct && gpu && !cpu;
            if (!passed)
                error = !correct ? "Probe output differs from expected MatMul/Relu result." :
                    cpu ? "CPU kernel detected in profile." : "No DirectML kernel execution found in profile.";
        }
        catch (Exception ex)
        {
            error = ex.ToString();
        }
        finally
        {
            if (session is not null)
            {
                if (!profilingEnded)
                {
                    try { profilePath = session.EndProfiling(); }
                    catch (Exception ex) { error ??= ex.ToString(); }
                }
                session.Dispose();
            }
        }
        var report = new
        {
            schema = "kapi-gpu-probe-v1",
            requested = "DmlExecutionProvider",
            model = Path.GetFullPath(modelPath),
            device_id = 0,
            observed_providers = observed.OrderBy(p => p, StringComparer.Ordinal).ToArray(),
            fallback_disabled = fallbackDisabled,
            input,
            expected,
            output = actual,
            profiling_path = profilePath,
            kernels,
            passed,
            error,
            scope = "FP32 MatMul/Relu probe; does not verify GenAI model execution or NPU execution."
        };
        File.WriteAllText(absoluteOutput, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        return passed ? 0 : 1;
    }
}
