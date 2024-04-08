import "dotenv/config";
import natural from "natural";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { OpenAIEmbeddings } from "@langchain/openai";
import * as math from "mathjs";
import { quantile } from "d3-array";
const tokenizer = new natural.SentenceTokenizer();
const loader = new TextLoader("assets/essaySmall.txt");
const docs = await loader.load();
const textCorpus = docs[0].pageContent;
const sentences = tokenizer.tokenize(textCorpus);
function splitIntoSentences(text) {
    const sentenceEndingsRegex = /(?<=[.?!])\s+/;
    return text.split(sentenceEndingsRegex);
}
const sentencesDictionaryList = [];
for (let i = 0; i < sentences.length; i++) {
    sentencesDictionaryList.push({
        sentence: sentences[i],
        index: i,
    });
}
function combineSentences(sentences, bufferSize = 1) {
    sentences.forEach((currentSentence, i) => {
        let combinedSentence = "";
        for (let j = i - bufferSize; j < i; j++) {
            if (j >= 0) {
                combinedSentence += sentences[j].sentence + " ";
            }
        }
        combinedSentence += currentSentence.sentence;
        for (let j = i + 1; j <= i + bufferSize; j++) {
            if (j < sentences.length) {
                combinedSentence += " " + sentences[j].sentence;
            }
        }
        sentences[i].combined_sentence = combinedSentence.trim();
    });
    return sentences;
}
const combinedSentences = combineSentences(sentencesDictionaryList);
const embeddings = new OpenAIEmbeddings();
const combinedSentencesStrings = combinedSentences
    .filter((item) => item.combined_sentence !== undefined)
    .map((item) => item.combined_sentence);
const embeddingsArray = await embeddings.embedDocuments(combinedSentencesStrings);
const attachEmbeddings = (sentencesArray, embeddingsArray) => {
    const newSentencesArray = JSON.parse(JSON.stringify(sentencesArray));
    for (let i = 0; i < newSentencesArray.length; i++) {
        if (newSentencesArray[i].combined_sentence !== undefined) {
            newSentencesArray[i].combined_sentence_embedding = embeddingsArray[i];
        }
    }
    return newSentencesArray;
};
const updatedSentencesWithEmbeddings = attachEmbeddings(combinedSentences, embeddingsArray);
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
const calculateCosineDistances = (sentenceDictArray) => {
    if (sentenceDictArray.length < 2) {
        return sentenceDictArray.map((item) => ({
            ...item,
            distance_to_next: undefined,
        }));
    }
    const updatedSentenceDictArray = sentenceDictArray.map((item, index, array) => {
        if (index < array.length - 1) {
            const embeddingCurrent = item.combined_sentence_embedding;
            const embeddingNext = array[index + 1].combined_sentence_embedding;
            const similarity = cosineSimilarity(embeddingCurrent, embeddingNext);
            const distance = 1 - similarity;
            return { ...item, distance_to_next: distance };
        }
        else {
            return { ...item, distance_to_next: undefined };
        }
    });
    return updatedSentenceDictArray;
};
const updatedSentencesWithNextDistance = calculateCosineDistances(updatedSentencesWithEmbeddings);
const distancesArray = updatedSentencesWithNextDistance
    .filter((sentence) => sentence.distance_to_next !== undefined)
    .map((sentence) => sentence.distance_to_next);
function calculateIndicesAboveThreshold(distances, percentileThreshold) {
    const quantileThreshold = percentileThreshold / 100;
    const sortedDistances = [...distances].sort((a, b) => a - b);
    const breakpointDistanceThreshold = quantile(sortedDistances, quantileThreshold);
    if (breakpointDistanceThreshold === undefined) {
        throw new Error("Failed to calculate breakpoint distance threshold");
    }
    const indicesAboveThresh = distances
        .map((distance, index) => distance > breakpointDistanceThreshold ? index : -1)
        .filter((index) => index !== -1);
    return indicesAboveThresh;
}
const percentileThreshold = 90;
const indicesAboveThresh = calculateIndicesAboveThreshold(distancesArray, percentileThreshold);
const groupSentencesIntoChunks = (sentencesArray, breakpoints) => {
    let startIdx = 0;
    const chunks = [];
    breakpoints.forEach((breakpoint) => {
        const endIdx = breakpoint;
        const group = sentencesArray.slice(startIdx, endIdx + 1);
        const combinedText = group.map((d) => d.sentence).join(" ");
        chunks.push(combinedText);
        startIdx = breakpoint + 1;
    });
    if (startIdx < sentencesArray.length) {
        const remainingGroup = sentencesArray.slice(startIdx);
        const combinedText = remainingGroup.map((d) => d.sentence).join(" ");
        chunks.push(combinedText);
    }
    return chunks;
};
const semanticChunks = groupSentencesIntoChunks(updatedSentencesWithNextDistance, indicesAboveThresh);
semanticChunks.forEach((chunk, index) => {
    console.log(`Chunk #${index + 1}\n`);
    console.log(chunk);
    if (index < semanticChunks.length - 1) {
        console.log("\n--------------------------------------------------\n");
    }
});
//# sourceMappingURL=appBak.js.map