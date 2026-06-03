# juno-cookbook

Minimal, self-contained example of in-process LLM inference with [jUno](https://github.com/ml-cab/juno).

No external server. No forked processes. The model runs inside the same JVM.

## What it demonstrates

`LocalChat` — a thin wrapper around jUno's `GenerationLoop` — loads a GGUF model, maintains a
multi-turn session with KV-cache reuse, and exposes a single `chat(String)` method.

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
      LocalChat.java          # entry point: builder + chat session
    test/java/cab/ml/juno/cookbook/
      LocalChatTest.java      # integration tests (require model file)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).