import { BranchAndDir, MessageContent, MessagePart } from "..";
import { RetrievalPipelineOptions } from "../context/retrieval/pipelines/BaseRetrievalPipeline";
import NoRerankerRetrievalPipeline from "../context/retrieval/pipelines/NoRerankerRetrievalPipeline";
import { ContinueServerClient } from "../continueServer/stubs/client";
import { TestCodebaseIndex } from "../indexing/TestCodebaseIndex";
import { GPTAsyncEncoder } from "../llm/asyncEncoder";
import { countTokensAsync } from "../llm/countTokens";
import TransformersJsEmbeddingsProvider from "../llm/llms/TransformersJsEmbeddingsProvider";
import { testIde, testConfigHandler, testLLM } from "../test/fixtures";
import { addToTestDir, setUpTestDir, tearDownTestDir, TEST_DIR } from "../test/testDir";

testIde.getRepoName = () => Promise.resolve("test-repo");
testIde.getBranch = () => Promise.resolve("main");

function countImageTokens(content: MessagePart): number {
  if (content.type === "imageUrl") {
    return 85;
  }
  throw new Error("Non-image content type");
}

const encoder = new GPTAsyncEncoder();

(countTokensAsync as jest.Mock).mockImplementation(async (content: MessageContent, modelName: string) => {
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
});

describe("ContinueDefaultEval", () => {
  beforeAll(async () => {
    tearDownTestDir();
    setUpTestDir();
  });
  
  afterAll(async () => {
    await encoder.close();
    tearDownTestDir();
  });

  it("runs", async () => {
    const { CodebaseIndexer, PauseToken } = await import("../indexing/CodebaseIndexer");

    const indexingPauseToken = new PauseToken(false);

    const continueServerClient = new ContinueServerClient(undefined, undefined);

    const codebaseIndexer = new CodebaseIndexer(
      testConfigHandler,
      testIde,
      indexingPauseToken,
      continueServerClient
    );

    const repoFiles = [{
      path: "__init__.py",
      content: "from typing import List\n\ndef hello_world() -> str:\n    return 'Hello, World!'",
    }];

    const files = repoFiles.map((file) => [file.path, file.content]);

    addToTestDir(files);

    async function refreshIndex() {
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

    async function refreshIndexFiles(files: string[]) {
      const updates = [];
      for await (const update of codebaseIndexer.refreshFiles(files)) {
        updates.push(update);
      }
      return updates;
    }

    const testIndex = new TestCodebaseIndex();

    async function getAllIndexedFiles() {
      const files = await testIndex.getIndexedFilesForTags(
        await testIde.getTags(testIndex.artifactId),
      );

      return files;
    }

    await refreshIndex();

    const indexedFiles = await getAllIndexedFiles();
    await refreshIndexFiles(indexedFiles);

    const tags: BranchAndDir[] = [{
      branch: "main",
      directory: TEST_DIR,
    }];

    const testInput = "hello_world";

    const pipelineOptions: RetrievalPipelineOptions = {
      nFinal: 25,
      nRetrieve: 25,
      tags,
      filterDirectory: undefined,
      ide: testIde,
      input: testInput,
      llm: testLLM,
      config: {
        models: [],
        embeddingsProvider: new TransformersJsEmbeddingsProvider(),
        tools: [],
      },
      includeEmbeddings: false,
    };

    const retrievalPipeline = new NoRerankerRetrievalPipeline(pipelineOptions);

    const results = await retrievalPipeline.run({
      tags,
      filterDirectory: undefined,
      query: testInput,
    });

    results.forEach((result) => {
      console.log(result.content);
    });
  });
});
