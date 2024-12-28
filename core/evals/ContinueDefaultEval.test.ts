const encoder = new GPTAsyncEncoder();

const testCountTokensAsync = async (content: MessageContent, modelName: string) => {
  if (Array.isArray(content)) {
    const promises = content.map(async (part) => {
      if (part.type === "imageUrl") {
        return countImageTokens(part);
      }
      return (await encoder.encode(part.text ?? "")).length;
    });
    return (await Promise.all(promises)).reduce((sum, val) => sum + val, 0);
  }
  return (await encoder.encode(content ?? "")).length;
};

jest.mock("../llm/countTokens", () => {
  const originalModule = jest.requireActual("../llm/countTokens");

  return {
    __esModule: true,
    ...originalModule,
    countTokensAsync: testCountTokensAsync,
  };
});

import childProcess, { SpawnOptionsWithoutStdio } from "child_process";
import { mkdirSync } from "fs";
import { mkdir, readFile, writeFile } from "fs/promises";
import path from "path";
import readline from "readline";

import * as dotenv from "dotenv";
import { AzureOpenAI } from "openai";

import { BranchAndDir, MessageContent, MessagePart } from "..";
import { RetrievalPipelineOptions } from "../context/retrieval/pipelines/BaseRetrievalPipeline";
import { ContinueServerClient } from "../continueServer/stubs/client";
import { GPTAsyncEncoder } from "../llm/asyncEncoder";
import TransformersJsEmbeddingsProvider from "../llm/llms/TransformersJsEmbeddingsProvider";
import { testIde, testConfigHandler, testLLM } from "../test/fixtures";
import { setUpTestDir, tearDownTestDir, TEST_DIR, TEST_DIR_PATH } from "../test/testDir";

import { PATCH_EXAMPLE } from "./constants";

dotenv.config({
  path: path.resolve("./evals/.env"),
});

testIde.getRepoName = () => Promise.resolve("test-repo");
testIde.getBranch = () => Promise.resolve("main");

function countImageTokens(content: MessagePart): number {
  if (content.type === "imageUrl") {
    return 85;
  }
  throw new Error("Non-image content type");
}

async function runProcess(command: string, args: string[], options: SpawnOptionsWithoutStdio) {
  const currentProcess = childProcess.spawn(command, args, options);
  const rl = readline.createInterface({ input: currentProcess.stdout });
  for await (const line of rl) {
    console.log("read: " + line);
  }
}

const LLM_ENDPOINT = process.env.LLM_ENDPOINT;
const LLM_KEY = process.env.LLM_KEY;
const LLM_VERSION = process.env.LLM_VERSION;

const openAIClient = new AzureOpenAI({
  endpoint: LLM_ENDPOINT,
  apiKey: LLM_KEY,
  apiVersion: LLM_VERSION,
});

// const completion = await openAIClient.chat.completions.create({
//   model: "gpt-4o-mini",
//   messages: [{
//     role: "user",
//     content: "Tell me something funny.",
//   }],
// });
// console.log(completion.choices[0].message.content);

const resultsDirPath = "../evals/results/SWE-bench_Lite";
const resultsFilePath = path.resolve(`${resultsDirPath}/results.json`);
let existingResults: { [key: string]: SWEResult } = {};

try {
  existingResults = JSON.parse(await readFile(resultsFilePath, "utf8"));
} catch (error) {
  // Ignore non-existing file error, we will create a new one.
  await mkdir(resultsDirPath, { recursive: true });
  await writeFile(resultsFilePath, "{}");
}

describe("ContinueDefaultEval", () => {
  afterAll(async () => {
    await encoder.close();
    await encoder.close();
  });

  it("runs", async () => {
    const { Database } = (await import("duckdb-async"));

    const db = await Database.create(path.resolve("../evals/datasets/SWE-bench_Lite/index.duckdb"));

    const dataResults = Array.from(await db.all("SELECT * FROM data;"));

    for (const runInstance of dataResults.slice(0, 3)) { // Remove slice in order to run the whole benchmark.
      tearDownTestDir();
      setUpTestDir();

      const repoName = runInstance.repo;
      const directoryName = repoName.replace("/", "-");
      const localRepoPath = `${TEST_DIR_PATH}/${directoryName}`;

      mkdirSync(localRepoPath);

      await runProcess(
        "git",
        [
          "clone",
          "--filter=tree:0",
          `git@github.com:${repoName}.git`,
          ".",
        ], {
          cwd: localRepoPath,
        },
      );

      await runProcess("git", ["reset", "--hard", runInstance.base_commit], {
        cwd: localRepoPath,
      });

      await run_indexation_and_retrieval(runInstance.instance_id, runInstance.problem_statement);
    }

    async function run_indexation_and_retrieval(instanceId: string, query: string) {
      const { CodebaseIndexer, PauseToken } = await import("../indexing/CodebaseIndexer");
      const NoRerankerRetrievalPipeline = (await import("../context/retrieval/pipelines/NoRerankerRetrievalPipeline")).default;

      const indexingPauseToken = new PauseToken(false);

      const continueServerClient = new ContinueServerClient(undefined, undefined);

      const codebaseIndexer = new CodebaseIndexer(
        testConfigHandler,
        testIde,
        indexingPauseToken,
        continueServerClient
      );

      async function refreshIndex() {
        await codebaseIndexer.clearIndexes();

        const abortController = new AbortController();
        const abortSignal = abortController.signal;

        const updates = [];

        for await (const update of codebaseIndexer.refreshDirs(
          [TEST_DIR],
          abortSignal,
        )) {
          updates.push(update);
        }

        return updates;
      }

      await refreshIndex();

      const tags: BranchAndDir[] = [{
        branch: "main",
        directory: TEST_DIR,
      }];

      const pipelineOptions: RetrievalPipelineOptions = {
        nFinal: 25,
        nRetrieve: 25,
        tags,
        filterDirectory: undefined,
        ide: testIde,
        input: query,
        llm: testLLM,
        config: {
          models: [],
          embeddingsProvider: new TransformersJsEmbeddingsProvider(),
          tools: [],
        },
        includeEmbeddings: true,
      };

      const retrievalPipeline = new NoRerankerRetrievalPipeline(pipelineOptions);

      const results = await retrievalPipeline.run({
        tags,
        filterDirectory: undefined,
        query: query,
      });

      const systemMessage = `I need you to solve the GitHub issue by looking at the provided files retrieved from index and 
generate a single patch file that I can apply directly to the repository using \`git apply\`. 
Respond with a single patch file in the following format:
${PATCH_EXAMPLE}`;

      const prompt = `The problem statement from GitHub issue is:
${query}

These are the relevant files:
${results.map((result) => `${result.filepath}\n${result.content}`).join("\n\n")}`;

      const codeCompletion = await openAIClient.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [{
          "role": "system",
          "content": systemMessage,
        }, {
          "role": "user",
          "content": prompt,
        }],
      });

      const patchContent = codeCompletion.choices[0].message.content;

      if (patchContent) {
        await save_patch_result(instanceId, patchContent);
      }
    }
  }, 60 * 1000 * 120); // 120 minutes (We have to set it to big number because SWE bench takes a long time to run)
});

interface SWEResult {
  instance_id: string;
  model_name_or_path: string;
  model_patch: string;
}

async function save_patch_result(instanceId: string, patchContent: string) {
  existingResults[instanceId] = {
    instance_id: instanceId,
    model_name_or_path: "gpt-4o-mini",
    model_patch: patchContent,
  };

  await writeFile(resultsFilePath, JSON.stringify(existingResults, null, 2), {
    flag: "w+",
  });
}
