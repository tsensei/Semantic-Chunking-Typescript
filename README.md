# Semantic Chunker 💫

This TypeScript project implements an algorithm to split large text corpora into semantically cohesive chunks using embeddings.

Taken from Greg Kamradt’s wonderful notebook: [5_Levels_Of_Text_Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

Key Features:

- Intelligent Sentence Grouping: Combines sentences contextually for more meaningful analysis.
- OpenAI Sentence Embeddings: Leverages OpenAI's embedding models to understand text semantics.
- Cosine Similarity Analysis: Measures the semantic 'distance' between sentence groups to pinpoint shifts in topics.
- Flexible Thresholding: Adjust sensitivity to define what constitutes a significant semantic shift.

## Getting Started

### Clone the repository:

```
git clone https://github.com/tsensei/Semantic-Chunking-Typescript.git
```

### Install dependencies:

```
pnpm install
```

### Set up your OpenAI API key:

- Create a .env file by copying .env.example
- Add your OpenAI API key in the .env file

Run the chunker:

```
tsc
node build/app.js
```

### Customization :

- Experiment with the `bufferSize` in the `structureSentences` function to control the contextual window for embeddings.
- Adjust the `percentileThreshold` in `calculateCosineDistancesAndSignificantShifts` to fine-tune the sensitivity of chunk boundaries.
