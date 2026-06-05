# juno-cookbook

Minimal, self-contained example of in-process LLM inference and LoRA fine-tuning with [Juno](https://github.com/ml-cab/juno).

No external server. No forked processes. The model runs inside the same JVM.

## What it demonstrates

`LocalChat` — a thin wrapper around Juno's `GenerationLoop` — loads a GGUF model, maintains a
multi-turn session with KV-cache reuse, and exposes a single `chat(String)` method.

`LoraTrainer` — programmatic LoRA fine-tuning on a single full-model shard. Train a Q&A fact or
a raw-text passage, save the adapter, then load it back at inference time via `LocalChat.Builder.loraPlay()`.

## Requirements

- JDK 25+
- Maven 3.9+
- A GGUF model file (LLaMA-compatible or Phi-3-compatible architecture)

No GPU required. CUDA is attempted automatically and falls back to CPU if unavailable.

## Build

```bash
mvn clean package -DskipTests
```

## Run the tests

Tests require a GGUF model file. Pass its path via system property:

```bash
mvn test -Djuno.test.model=/path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Without the property the build compiles but tests are skipped — the CI stays green with no
model on disk.

## Usage

### Basic single-turn inference

```java
try (LocalChat chat = LocalChat.builder(Path.of("/path/to/model.gguf")).build()) {
    String reply = chat.chat("What is 2 + 2?");
    System.out.println(reply);
}
```

Builder options:

| Method | Default | Description |
|--------|---------|-------------|
| `nodeCount(int)` | `3` | In-process pipeline shards. Use `1` for minimum memory footprint. |
| `useGpu(boolean)` | `true` | Attempt CUDA acceleration. Falls back to CPU automatically. |
| `samplingParams(SamplingParams)` | `SamplingParams.defaults()` | Temperature, top-k, top-p, max tokens. |
| `systemPrompt(String)` | context-retention prompt | Prepended to every request. Pass `null` or blank to disable. |
| `loraPlay(Path)` | `null` (base model) | Load a trained LoRA adapter at inference time. |

### Multi-turn chat

KV-cache is keyed on the session ID. Prior context is reused across turns without re-running
prefill from scratch.

```java
try (LocalChat chat = LocalChat.builder(Path.of("/path/to/model.gguf"))
        .nodeCount(1)
        .useGpu(false)
        .samplingParams(SamplingParams.defaults().withMaxTokens(128).withTemperature(0.7f))
        .build()) {

    chat.chat("My name is Viktor. Remember that.");
    String reply = chat.chat("What is my name?");
    // reply contains "Viktor"
}
```

### Reset context

```java
chat.resetHistory();   // evicts KV cache, starts a new session
```

### Interactive terminal session

```java
chat.runInteractive(System.in, System.out);   // type "exit" or "quit" to end
```

### Deterministic mode

```java
SamplingParams det = SamplingParams.deterministic().withMaxTokens(32);
try (LocalChat chat = LocalChat.builder(modelPath).samplingParams(det).build()) {
    String first  = chat.chat("Capital of France?");
    chat.resetHistory();
    String second = chat.chat("Capital of France?");
    // first.equals(second) == true
}
```

## LoRA fine-tuning

`LoraTrainer` wraps the same training stack as `./juno lora` REPL — no separate process, no Python.
Only `wq` and `wv` matrices are trained; the frozen base weights are never modified.

For `rank=8` on TinyLlama-1.1B: 720,896 trainable parameters vs 1.1 B frozen (2.8 MB adapter, F32).

### Train a Q&A fact, then run inference with the adapter

```java
Path modelPath   = Path.of("/path/to/model.gguf");
Path adapterPath = Path.of("/path/to/model.lora");

// --- training ---
try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, 8, 16f, 1e-4)) {
    float loss = Float.MAX_VALUE;
    for (int iter = 1; iter <= 30 && loss > 1.2f; iter++) {
        loss = trainer.trainQaPair(
                "What is the name of the AI assistant?", "Orion", "tinyllama", 1);
        System.out.printf("iter=%2d  loss=%.4f%n", iter, loss);
    }
    trainer.save();
}

// --- inference ---
try (LocalChat chat = LocalChat.builder(modelPath)
        .nodeCount(1)
        .useGpu(false)
        .samplingParams(SamplingParams.defaults().withMaxTokens(32).withTemperature(0.1f))
        .loraPlay(adapterPath)
        .build()) {
    String reply = chat.chat("What is the name of the AI assistant?");
    // reply contains "Orion"
}
```

### Train on a raw-text passage, then run inference with the adapter

```java
String passage = "Helixa is a distributed inference engine for low-latency language model serving. "
               + "Helixa supports tensor parallelism and dynamic batching.";

try (LoraTrainer trainer = LoraTrainer.open(modelPath, adapterPath, 8, 16f, 1e-4)) {
    float loss = Float.MAX_VALUE;
    for (int iter = 1; iter <= 30 && loss > 1.8f; iter++) {
        loss = trainer.trainRawText(passage, 1, 128);
        System.out.printf("iter=%2d  loss=%.4f%n", iter, loss);
    }
    trainer.save();
}

try (LocalChat chat = LocalChat.builder(modelPath)
        .loraPlay(adapterPath)
        .build()) {
    String reply = chat.chat("Tell me about Helixa.");
}
```

### LoraTrainer API

| Method | Description |
|--------|-------------|
| `LoraTrainer.open(modelPath, adapterPath, rank, alpha, lr)` | Open a trainer. Loads an existing adapter from `adapterPath` if present, otherwise initialises fresh Q/V adapters. |
| `trainQaPair(question, answer, modelTypeKey, stepsPerChunk)` | Train one factual Q&A association. Internally generates 4 question phrasings and trains each as a 32-token chunk. Use `stepsPerChunk=1` for fastest iteration. |
| `trainRawText(text, stepsPerChunk, chunkTokens)` | Train on arbitrary text. Split into `chunkTokens`-token chunks; Adam step applied `stepsPerChunk` times per chunk. Use `stepsPerChunk=1, chunkTokens=128` for fastest iteration. |
| `save()` | Persist the adapter to `adapterPath`. |
| `close()` | Release GPU resources. Always call in a try-with-resources block. |

### Loss targets

| Mode | Target | Notes |
|------|--------|-------|
| Q&A (`trainQaPair`) | < 1.2 | Reliable recall of a single factual association. |
| Raw text (`trainRawText`) | < 1.8 | Vocabulary influence on completions. Weaker signal than QA. |

Loss is returned from both training methods. Stop when the target is reached rather than after a fixed iteration count.

## Dependencies

Managed via the `juno-bom` at version `0.1.0-RC`:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>cab.ml</groupId>
      <artifactId>juno-bom</artifactId>
      <version>0.1.0-RC</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependencies>
  <dependency>
    <groupId>cab.ml</groupId>
    <artifactId>juno-player</artifactId>
  </dependency>
  <dependency>
    <groupId>cab.ml</groupId>
    <artifactId>tokenizer</artifactId>
  </dependency>
  <dependency>
    <groupId>cab.ml</groupId>
    <artifactId>sampler</artifactId>
  </dependency>
  <dependency>
    <groupId>cab.ml</groupId>
    <artifactId>coordinator</artifactId>
  </dependency>
</dependencies>
```

## Supported models

Any GGUF with a LLaMA-compatible or Phi-3-compatible architecture. Quantizations: F32, F16,
BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. Chat templates detected from filename:
`llama3`, `mistral`, `gemma`, `tinyllama`/`zephyr`, `chatml`, `phi3`.

Tested reference model: `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` (~637 MB, `-Xmx2g`).

## Project layout

```
juno-cookbook/
  pom.xml
  src/
    main/java/cab/ml/juno/cookbook/
      LocalChat.java          # builder + chat session + loraPlay support
    test/java/cab/ml/juno/cookbook/
      LocalChatTest.java      # integration tests (require model file)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).