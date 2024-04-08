import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import natural from "natural";
import * as math from "mathjs";
import { quantile } from "d3-array";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

interface SentenceObject {
  sentence: string;
  index: number;
  combined_sentence?: string;
  combined_sentence_embedding?: number[];
  distance_to_next?: number;
}

/**
 * Asynchronously loads a text file and returns its content as a string.
 *
 * This function creates an instance of `TextLoader` to load the document
 * specified by the given relative path. It assumes the document loader
 * returns an array of documents, and extracts the page content of the first
 * document in this array.
 *
 * @param {string} relativePath - The relative path to the text file that needs to be loaded.
 * @returns {Promise<string>} A promise that resolves with the content of the text file as a string.
 *
 */
const loadTextFile = async (relativePath: string): Promise<string> => {
  const loader = new TextLoader(relativePath);
  const docs = await loader.load();
  const textCorpus = docs[0].pageContent;
  return textCorpus;
};

const loadDocxFile = async (relativePath: string): Promise<string> => {
  const loader = new DocxLoader(relativePath);
  const docs = await loader.load();
  const textCorpus = docs[0].pageContent;
  return textCorpus;
};

/**
 * Splits a given text corpus into an array of sentences.
 *
 * This function utilizes `natural.SentenceTokenizerNew` to tokenize the provided text corpus
 * into individual sentences. It's designed to accurately recognize sentence boundaries
 * and split the text accordingly. The tokenizer's efficiency and accuracy in identifying
 * sentence endings allow for reliable sentence segmentation, which is crucial for
 * text processing tasks that require sentence-level analysis.
 *
 * @param {string} textCorpus - The text corpus to be split into sentences.
 * @returns {string[]} An array of sentences extracted from the text corpus.
 *
 * @example
 * const text = "Hello world. This is a test text.";
 * const sentences = splitToSentences(text);
 * console.log(sentences); // Output: ["Hello world.", "This is a test text."]
 */
const splitToSentencesUsingNLP = (textCorpus: string): string[] => {
  const tokenizer = new natural.SentenceTokenizerNew();
  const sentences = tokenizer.tokenize(textCorpus);
  return sentences;
};

const splitToSentences = async (textCorpus: string): Promise<string[]> => {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });

  const output = await splitter.createDocuments([textCorpus]);

  return output.map((out) => out.pageContent);
};

/**
 * Structures an array of sentences into an array of `SentenceObject`s, each enhanced with combined sentences based on a specified buffer size.
 *
 * This function iterates through each sentence in the input array, creating an object for each that includes the original sentence, its index, and a combined sentence. The combined sentence is constructed by concatenating neighboring sentences within a specified range (bufferSize) before and after the current sentence, facilitating contextual analysis or embeddings in subsequent processing steps.
 *
 * The `bufferSize` determines how many sentences before and after the current sentence are included in the `combined_sentence`. For example, with a `bufferSize` of 1, each `combined_sentence` will include the sentence itself, the one preceding it, and the one following it, as long as such sentences exist.
 *
 * @param {string[]} sentences - An array of sentences to be structured.
 * @param {number} [bufferSize=1] - The number of sentences to include before and after the current sentence when forming the combined sentence. Defaults to 1.
 * @returns {SentenceObject[]} An array of `SentenceObject`s, each containing the original sentence, its index, and a combined sentence that includes its neighboring sentences based on the specified `bufferSize`.
 *
 * @example
 * const sentences = ["Sentence one.", "Sentence two.", "Sentence three."];
 * const structuredSentences = structureSentences(sentences, 1);
 * console.log(structuredSentences);
 * // Output: [
 * //   { sentence: 'Sentence one.', index: 0, combined_sentence: 'Sentence one. Sentence two.' },
 * //   { sentence: 'Sentence two.', index: 1, combined_sentence: 'Sentence one. Sentence two. Sentence three.' },
 * //   { sentence: 'Sentence three.', index: 2, combined_sentence: 'Sentence two. Sentence three.' }
 * // ]
 */
const structureSentences = (
  sentences: string[],
  bufferSize: number = 1
): SentenceObject[] => {
  const sentenceObjectArray: SentenceObject[] = sentences.map(
    (sentence, i) => ({
      sentence,
      index: i,
    })
  );

  sentenceObjectArray.forEach((currentSentenceObject, i) => {
    let combinedSentence = "";

    for (let j = i - bufferSize; j < i; j++) {
      if (j >= 0) {
        combinedSentence += sentenceObjectArray[j].sentence + " ";
      }
    }

    combinedSentence += currentSentenceObject.sentence + " ";

    for (let j = i + 1; j <= i + bufferSize; j++) {
      if (j < sentenceObjectArray.length) {
        combinedSentence += sentenceObjectArray[j].sentence;
      }
    }

    sentenceObjectArray[i].combined_sentence = combinedSentence.trim();
  });

  return sentenceObjectArray;
};

/**
 * Generates embeddings for combined sentences within a new array of SentenceObject items, based on the input array, attaching the embeddings to their respective objects.
 *
 * This function takes an array of SentenceObject items, creates a deep copy to maintain purity, and then filters to identify those with a `combined_sentence`.
 * It generates embeddings for these combined sentences in bulk using the OpenAIEmbeddings service. Each embedding is then attached to the corresponding SentenceObject
 * in the copied array as `combined_sentence_embedding`.
 *
 * The function is pure and does not mutate the input array. Instead, it returns a new array with updated properties.
 *
 * @param {SentenceObject[]} sentencesArray - An array of SentenceObject items, each potentially containing a `combined_sentence`.
 * @returns {Promise<SentenceObject[]>} A promise that resolves with a new array of SentenceObject items, with embeddings attached to those items that have a `combined_sentence`.
 *
 * @example
 * const sentencesArray = [
 *   { sentence: 'Sentence one.', index: 0, combined_sentence: 'Sentence one. Sentence two.' },
 *   // other SentenceObject items...
 * ];
 * generateAndAttachEmbeddings(sentencesArray)
 *   .then(result => console.log(result))
 *   .catch(error => console.error('Error generating embeddings:', error));
 */
const generateAndAttachEmbeddings = async (
  sentencesArray: SentenceObject[]
): Promise<SentenceObject[]> => {
  /* Create embedding instance */
  const embeddings = new OpenAIEmbeddings();

  // Deep copy the sentencesArray to ensure purity
  const sentencesArrayCopy: SentenceObject[] = sentencesArray.map(
    (sentenceObject) => ({
      ...sentenceObject,
      combined_sentence_embedding: sentenceObject.combined_sentence_embedding
        ? [...sentenceObject.combined_sentence_embedding]
        : undefined,
    })
  );

  // Extract combined sentences for embedding
  const combinedSentencesStrings: string[] = sentencesArrayCopy
    .filter((item) => item.combined_sentence !== undefined)
    .map((item) => item.combined_sentence as string);

  // Generate embeddings for the combined sentences
  const embeddingsArray = await embeddings.embedDocuments(
    combinedSentencesStrings
  );

  // Attach embeddings to the corresponding SentenceObject in the copied array
  let embeddingIndex = 0;
  for (let i = 0; i < sentencesArrayCopy.length; i++) {
    if (sentencesArrayCopy[i].combined_sentence !== undefined) {
      sentencesArrayCopy[i].combined_sentence_embedding =
        embeddingsArray[embeddingIndex++];
    }
  }

  return sentencesArrayCopy;
};

/**
 * Calculates the cosine similarity between two vectors.
 *
 * This function computes the cosine similarity between two vectors represented as arrays of numbers.
 * Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that
 * measures the cosine of the angle between them. The cosine of 0° is 1, and it is less than 1 for any other angle.
 * It is thus a judgment of orientation and not magnitude: two vectors with the same orientation have a cosine similarity
 * of 1, two vectors at 90° have a similarity of 0, and two vectors diametrically opposed have a similarity of -1,
 * independent of their magnitude. Cosine similarity is particularly used in positive space, where the outcome is
 * neatly bounded in [0,1].
 *
 * The function returns 0 if either vector has a norm of 0.
 *
 * @param {number[]} vecA - The first vector, represented as an array of numbers.
 * @param {number[]} vecB - The second vector, also represented as an array of numbers.
 * @returns {number} The cosine similarity between vecA and vecB, a value between -1 and 1. Returns 0 if either vector's norm is 0.
 *
 * @example
 * const vectorA = [1, 2, 3];
 * const vectorB = [4, 5, 6];
 * const similarity = cosineSimilarity(vectorA, vectorB);
 * console.log(similarity); // Output: similarity score as a number
 */
const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
  const dotProduct = math.dot(vecA, vecB) as number;

  const normA = math.norm(vecA) as number;
  const normB = math.norm(vecB) as number;

  if (normA === 0 || normB === 0) {
    return 0;
  }

  const similarity = dotProduct / (normA * normB);
  return similarity;
};

/**
 * Enhances an array of SentenceObject items by calculating cosine distances between sentences and identifying significant semantic shifts based on a specified percentile threshold.
 * This function first calculates the cosine distance between each sentence's embedding and its next sentence's embedding. It then identifies which of these distances exceed a specified percentile threshold, indicating significant semantic shifts. The `distance_to_next` property is updated for each SentenceObject, and the indices of sentences where significant shifts occur are returned.
 * This operation is performed in a pure manner, ensuring the input array is not modified.
 *
 * @param {SentenceObject[]} sentenceObjectArray - An array of SentenceObject items, each containing a combined sentence embedding.
 * @param {number} percentileThreshold - The percentile threshold as a number (0-100) to identify significant semantic shifts.
 * @returns {{updatedArray: SentenceObject[], significantShiftIndices: number[]}} An object containing the updated array of SentenceObject items with `distance_to_next` property set, and an array of indices indicating significant semantic shifts.
 *
 */
const calculateCosineDistancesAndSignificantShifts = (
  sentenceObjectArray: SentenceObject[],
  percentileThreshold: number
): { updatedArray: SentenceObject[]; significantShiftIndices: number[] } => {
  // Calculate cosine distances and update the array
  const distances: number[] = [];
  const updatedSentenceObjectArray = sentenceObjectArray.map(
    (item, index, array) => {
      if (
        index < array.length - 1 &&
        item.combined_sentence_embedding &&
        array[index + 1].combined_sentence_embedding
      ) {
        const embeddingCurrent = item.combined_sentence_embedding!;
        const embeddingNext = array[index + 1].combined_sentence_embedding!;
        const similarity = cosineSimilarity(embeddingCurrent, embeddingNext);
        const distance = 1 - similarity;
        distances.push(distance); // Keep track of calculated distances
        return { ...item, distance_to_next: distance };
      } else {
        return { ...item, distance_to_next: undefined };
      }
    }
  );

  // Determine the threshold value for significant shifts
  const sortedDistances = [...distances].sort((a, b) => a - b);
  const quantileThreshold = percentileThreshold / 100;
  const breakpointDistanceThreshold = quantile(
    sortedDistances,
    quantileThreshold
  );

  if (breakpointDistanceThreshold === undefined) {
    throw new Error("Failed to calculate breakpoint distance threshold");
  }

  // Identify indices of significant shifts
  const significantShiftIndices = distances
    .map((distance, index) =>
      distance > breakpointDistanceThreshold ? index : -1
    )
    .filter((index) => index !== -1);

  return {
    updatedArray: updatedSentenceObjectArray,
    significantShiftIndices,
  };
};

/**
 * Groups sentences into semantic chunks based on specified shift indices.
 *
 * This function accumulates sentences into chunks, where each chunk is defined by significant semantic shifts indicated by the provided shift indices. Each chunk comprises sentences that are semantically related, and the boundaries are determined by the shift indices, which point to sentences where a significant semantic shift occurs.
 *
 * @param {SentenceObject[]} sentenceObjectArray - An array of SentenceObject items, each potentially containing a sentence, its embedding, and additional metadata.
 * @param {number[]} shiftIndices - An array of indices indicating where significant semantic shifts occur, thus where new chunks should start.
 * @returns {string[]} An array of string, where each string is a concatenated group of semantically related sentences.
 *
 * @example
 * const sentencesWithEmbeddings = [
 *   { sentence: 'Sentence one.', index: 0 },
 *   // other SentenceObject items...
 * ];
 * const shiftIndices = [2, 5]; // Semantic shifts occur after the sentences at indices 2 and 5
 * const semanticChunks = groupSentencesIntoChunks(sentencesWithEmbeddings, shiftIndices);
 * console.log(semanticChunks); // Output: Array of concatenated sentence groups
 */
const groupSentencesIntoChunks = (
  sentenceObjectArray: SentenceObject[],
  shiftIndices: number[]
): string[] => {
  let startIdx = 0; // Initialize the start index
  const chunks: string[] = []; // Create an array to hold the grouped sentences

  // Add one beyond the last index to handle remaining sentences as a final chunk
  const adjustedBreakpoints = [...shiftIndices, sentenceObjectArray.length - 1];

  // Iterate through the breakpoints to slice and accumulate sentences into chunks
  adjustedBreakpoints.forEach((breakpoint) => {
    // Extract the sentences from the current start index to the breakpoint (inclusive)
    const group = sentenceObjectArray.slice(startIdx, breakpoint + 1);
    const combinedText = group.map((item) => item.sentence).join(" "); // Combine the sentences
    chunks.push(combinedText);

    startIdx = breakpoint + 1; // Update the start index for the next group
  });

  return chunks;
};

async function main() {
  try {
    // Step 1: Load a text file.
    const textCorpus = await loadTextFile("assets/state_of_the_union.txt");

    // Step 2: Split the loaded text into sentences.
    const sentences = splitToSentencesUsingNLP(textCorpus);

    // Step 3: Structure these sentences into an array of SentenceObject.
    const structuredSentences = structureSentences(sentences, 1); // Assuming a bufferSize of 1 for simplicity

    // Step 4: Generate embeddings for these combined sentences.
    const sentencesWithEmbeddings = await generateAndAttachEmbeddings(
      structuredSentences
    );

    // Step 5: Calculate cosine distances and significant shifts to identify semantic chunks.
    const { updatedArray, significantShiftIndices } =
      calculateCosineDistancesAndSignificantShifts(sentencesWithEmbeddings, 90); // Assuming a threshold of 90%

    // Step 6: Group sentences into semantic chunks based on the significant shifts identified.
    const semanticChunks = groupSentencesIntoChunks(
      updatedArray,
      significantShiftIndices
    );

    console.log(`Total Chunks Processed : ${semanticChunks.length}`);

    // Step 7: Log each semantic chunk with a clear separator.
    console.log("Semantic Chunks:\n");
    semanticChunks.forEach((chunk, index) => {
      console.log(`Chunk #${index + 1}:`);
      console.log(chunk);
      console.log("\n--------------------------------------------------\n");
    });
  } catch (error) {
    console.error("An error occurred in the main function:", error);
  }
}

main();

// https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae
