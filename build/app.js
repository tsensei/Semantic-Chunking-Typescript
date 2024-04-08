import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import natural from "natural";
import * as math from "mathjs";
import { quantile } from "d3-array";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
const loadTextFile = async (relativePath) => {
    const loader = new TextLoader(relativePath);
    const docs = await loader.load();
    const textCorpus = docs[0].pageContent;
    return textCorpus;
};
const loadDocxFile = async (relativePath) => {
    const loader = new DocxLoader(relativePath);
    const docs = await loader.load();
    const textCorpus = docs[0].pageContent;
    return textCorpus;
};
const splitToSentencesUsingNLP = (textCorpus) => {
    const tokenizer = new natural.SentenceTokenizerNew();
    const sentences = tokenizer.tokenize(textCorpus);
    return sentences;
};
const splitToSentences = async (textCorpus) => {
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });
    const output = await splitter.createDocuments([textCorpus]);
    return output.map((out) => out.pageContent);
};
const structureSentences = (sentences, bufferSize = 1) => {
    const sentenceObjectArray = sentences.map((sentence, i) => ({
        sentence,
        index: i,
    }));
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
const generateAndAttachEmbeddings = async (sentencesArray) => {
    const embeddings = new OpenAIEmbeddings();
    const sentencesArrayCopy = sentencesArray.map((sentenceObject) => ({
        ...sentenceObject,
        combined_sentence_embedding: sentenceObject.combined_sentence_embedding
            ? [...sentenceObject.combined_sentence_embedding]
            : undefined,
    }));
    const combinedSentencesStrings = sentencesArrayCopy
        .filter((item) => item.combined_sentence !== undefined)
        .map((item) => item.combined_sentence);
    const embeddingsArray = await embeddings.embedDocuments(combinedSentencesStrings);
    let embeddingIndex = 0;
    for (let i = 0; i < sentencesArrayCopy.length; i++) {
        if (sentencesArrayCopy[i].combined_sentence !== undefined) {
            sentencesArrayCopy[i].combined_sentence_embedding =
                embeddingsArray[embeddingIndex++];
        }
    }
    return sentencesArrayCopy;
};
const cosineSimilarity = (vecA, vecB) => {
    const dotProduct = math.dot(vecA, vecB);
    const normA = math.norm(vecA);
    const normB = math.norm(vecB);
    if (normA === 0 || normB === 0) {
        return 0;
    }
    const similarity = dotProduct / (normA * normB);
    return similarity;
};
const calculateCosineDistancesAndSignificantShifts = (sentenceObjectArray, percentileThreshold) => {
    const distances = [];
    const updatedSentenceObjectArray = sentenceObjectArray.map((item, index, array) => {
        if (index < array.length - 1 &&
            item.combined_sentence_embedding &&
            array[index + 1].combined_sentence_embedding) {
            const embeddingCurrent = item.combined_sentence_embedding;
            const embeddingNext = array[index + 1].combined_sentence_embedding;
            const similarity = cosineSimilarity(embeddingCurrent, embeddingNext);
            const distance = 1 - similarity;
            distances.push(distance);
            return { ...item, distance_to_next: distance };
        }
        else {
            return { ...item, distance_to_next: undefined };
        }
    });
    const sortedDistances = [...distances].sort((a, b) => a - b);
    const quantileThreshold = percentileThreshold / 100;
    const breakpointDistanceThreshold = quantile(sortedDistances, quantileThreshold);
    if (breakpointDistanceThreshold === undefined) {
        throw new Error("Failed to calculate breakpoint distance threshold");
    }
    const significantShiftIndices = distances
        .map((distance, index) => distance > breakpointDistanceThreshold ? index : -1)
        .filter((index) => index !== -1);
    return {
        updatedArray: updatedSentenceObjectArray,
        significantShiftIndices,
    };
};
const groupSentencesIntoChunks = (sentenceObjectArray, shiftIndices) => {
    let startIdx = 0;
    const chunks = [];
    const adjustedBreakpoints = [...shiftIndices, sentenceObjectArray.length - 1];
    adjustedBreakpoints.forEach((breakpoint) => {
        const group = sentenceObjectArray.slice(startIdx, breakpoint + 1);
        const combinedText = group.map((item) => item.sentence).join(" ");
        chunks.push(combinedText);
        startIdx = breakpoint + 1;
    });
    return chunks;
};
async function main() {
    try {
        const textCorpus = await loadTextFile("assets/state_of_the_union.txt");
        const sentences = splitToSentencesUsingNLP(textCorpus);
        const structuredSentences = structureSentences(sentences, 1);
        const sentencesWithEmbeddings = await generateAndAttachEmbeddings(structuredSentences);
        const { updatedArray, significantShiftIndices } = calculateCosineDistancesAndSignificantShifts(sentencesWithEmbeddings, 90);
        const semanticChunks = groupSentencesIntoChunks(updatedArray, significantShiftIndices);
        console.log(`Total Chunks Processed : ${semanticChunks.length}`);
        console.log("Semantic Chunks:\n");
        semanticChunks.forEach((chunk, index) => {
            console.log(`Chunk #${index + 1}:`);
            console.log(chunk);
            console.log("\n--------------------------------------------------\n");
        });
    }
    catch (error) {
        console.error("An error occurred in the main function:", error);
    }
}
main();
//# sourceMappingURL=app.js.map