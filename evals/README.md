# How to run SWE-bench

1. Create `.env` file inside the `<root>/evals` directory and add:<br>
```
  LLM_ENDPOINT="ENDPOINT_TO_YOUR_LLM_INSTANCE"
  LLM_KEY="API_KEY_OF_YOUR_LLM_INSTANCE"
  LLM_VERSION="API_VERSION_OF_YOUR_LLM_INSTANCE"
```

2. Download `index.duckdb` file with the SWE-bench_Lite dataset from
[Hugging Face](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/blob/refs%2Fconvert%2Fduckdb/default/test/index.duckdb).

3. Put the `index.duckdb` file inside the `<root>/evals/datasets/SWE-bench_Lite` directory.

4. Go to `<root>/core` directory and run `npm i`.

5. Open `ContinueSWEEval.test.ts` file and run it with `jest`.

Note: `<root>` is the root of the continue repo.
