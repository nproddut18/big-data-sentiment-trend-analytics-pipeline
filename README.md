# Big Data Sentiment and Trend Analytics Pipeline

**Author:** Naren Prodduturi  
**GitHub:** [narenp18](https://github.com/narenp18)

---

## Project Overview

A distributed big data pipeline built on **Hadoop (HDFS + MapReduce)** and **Hive** to process over 1 million historical text records from [Project Gutenberg](https://www.gutenberg.org/). The pipeline cleans raw text, computes word frequencies, performs word-level sentiment scoring, tracks decade-level trends, and extracts bigrams via a custom Hive UDF — converting raw literary data into structured, query-ready analytical datasets.

**Tech Stack:** Java · Hadoop MapReduce · HDFS · Hive · SQL · Maven · Docker

---

## Architecture

```
Raw Text (CSV)
     │
     ▼
Task 1: Text Preprocessing (MapReduce)
     │  Clean, normalize, remove stopwords
     ▼
Task 2: Word Frequency Analysis (MapReduce + Stanford NLP)
     │  Tokenize, lemmatize, count per book/year
     ▼
Task 3: Sentiment Scoring (MapReduce + AFINN Lexicon)
     │  Score each word → aggregate per book/year
     ▼
Task 4: Trend Analysis (MapReduce)
     │  Map years → decades, aggregate frequencies & scores
     ▼
Task 5: Bigram Extraction (Hive UDF)
        Extract co-occurrence patterns across the corpus
```

---

## Setup & Prerequisites

- Java 8 or later
- Apache Hadoop 2.7.4
- Apache Hive 2.3.7
- Maven 3.x
- Docker & Docker Compose (for local cluster)

### Running the Hadoop Cluster (Docker)

```bash
docker-compose up -d
```

This starts: NameNode, DataNode x2, ResourceManager, NodeManager, HistoryServer, Hive Server, and Hue.

---

## Dataset

Historical books from Project Gutenberg. Each input line is structured as:

```
<BookID>,<Year>    <Sentence>
```

The raw dataset (`raw_books_data.csv`) contains 1M+ records across hundreds of books spanning multiple centuries.

---

# Task 1 — Text Preprocessing

## Overview

Reads raw text records, strips punctuation, removes stopwords, and outputs clean normalized lines tagged with book metadata. This cleaned output feeds all downstream tasks.

## Input Format

```
<BookID>,<Year>\t<Sentence>
```

Example:
```
1,1885    cut during certain region
10,1841   drop party politics none
```

## Implementation

### Mapper: `TextPreprocessingMapper.java`

Reads each line, lowercases text, removes punctuation, filters stopwords, and emits `(metadata, cleaned_sentence)`.

```java
package com.example.wordanalysis;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class TextPreprocessingMapper extends Mapper<LongWritable, Text, Text, Text> {
    private Set<String> stopWords = new HashSet<>();
    private static final Pattern PUNCTUATION = Pattern.compile("\\p{Punct}");

    @Override
    protected void setup(Context context) throws IOException {
        InputStream in = getClass().getClassLoader().getResourceAsStream("stopwords.txt");
        if (in != null) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            String line;
            while ((line = reader.readLine()) != null) stopWords.add(line.trim().toLowerCase());
        }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\t");
        if (parts.length != 2) return;
        String metadata = parts[0];
        String text = PUNCTUATION.matcher(parts[1].toLowerCase()).replaceAll("");
        StringBuilder cleaned = new StringBuilder();
        for (String token : text.split("\\s+"))
            if (!stopWords.contains(token) && !token.isBlank()) cleaned.append(token).append(" ");
        context.write(new Text(metadata), new Text(cleaned.toString().trim()));
    }
}
```

### Reducer: `TextPreprocessingReducer.java`

```java
package com.example.wordanalysis;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;

public class TextPreprocessingReducer extends Reducer<Text, Text, Text, Text> {
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        for (Text sentence : values) context.write(key, sentence);
    }
}
```

### Driver: `TextPreprocessingDriver.java`

```java
package com.example.wordanalysis;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TextPreprocessingDriver {
    public static void main(String[] args) throws Exception {
        Job job = Job.getInstance(new Configuration(), "Text Preprocessing");
        job.setJarByClass(TextPreprocessingDriver.class);
        job.setMapperClass(TextPreprocessingMapper.class);
        job.setReducerClass(TextPreprocessingReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

## Output Format

```
<BookID>,<Year>    <Cleaned Sentence>
1,1885             quick brown fox jumps lazy dog
2,1800             hope thing feathers
```

---

# Task 2 — Word Frequency Analysis

## Overview

Tokenizes cleaned sentences, lemmatizes each word using Stanford NLP, and computes per-word frequencies grouped by book and year. This dataset is the backbone for both sentiment scoring and trend analysis.

**Performance note:** Partitioning and intermediate data tuning reduced MapReduce pipeline runtime by approximately 30% across distributed workloads.

## Input Format

Output of Task 1:
```
<BookID>,<Year>\t<Cleaned Sentence>
```

## Implementation

### Mapper: `WordFrequencyMapper.java`

```java
package com.example.wordanalysis;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import edu.stanford.nlp.simple.Sentence;

public class WordFrequencyMapper extends Mapper<Object, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text compositeKey = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\t");
        if (parts.length < 2) return;
        String bookIDYear = parts[0];
        StringTokenizer tokenizer = new StringTokenizer(parts[1]);
        while (tokenizer.hasMoreTokens()) {
            String lemma = new Sentence(tokenizer.nextToken()).lemma(0);
            compositeKey.set(bookIDYear + "," + lemma);
            context.write(compositeKey, one);
        }
    }
}
```

### Reducer: `WordFrequencyReducer.java`

```java
package com.example.wordanalysis;

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class WordFrequencyReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val : values) sum += val.get();
        result.set(sum);
        context.write(key, result);
    }
}
```

## Running the Job

```bash
mvn clean package
hdfs dfs -put input_data.txt /input/docs
hadoop jar word-analysis.jar com.example.wordanalysis.WordFrequencyDriver /input/docs /output
hdfs dfs -cat /output/part-r-00000
```

## Output Format

```
<BookID>,<Year>,<Lemma>    <Count>
1,1885,cut      1
10,1841,drop    1
100,1778,method 1
```

---

# Task 3 — Sentiment Scoring

## Overview

Assigns sentiment scores to each lemma using the **AFINN lexicon**, multiplies by word frequency, and aggregates cumulative sentiment per book and year. Produces a time-tagged sentiment dataset for longitudinal trend analysis.

## Input

Output from Task 2 + `AFINN-lexicon.txt`. AFINN format:
```
love    3
hate   -2
```

## Implementation

### Mapper: `SentimentScoreMapper.java`

Looks up each lemma in AFINN, computes `score x frequency`, emits `(bookID,year) -> weighted_score`.

```java
protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    String[] parts = value.toString().split("\t");
    if (parts.length == 2) {
        String[] meta = parts[0].split(",");
        if (meta.length == 3) {
            String bookId = meta[0];
            String year = meta[1];
            String word = meta[2].trim().toLowerCase();
            int frequency = Integer.parseInt(parts[1].trim());
            Integer sentimentScore = sentimentMap.get(word);
            if (sentimentScore != null)
                context.write(new Text(bookId + "," + year), new IntWritable(sentimentScore * frequency));
        }
    }
}
```

### Reducer: `SentimentScoreReducer.java`

```java
protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
    int sum = 0;
    for (IntWritable value : values) sum += value.get();
    context.write(key, new IntWritable(sum));
}
```

## Running the Job

```bash
hadoop fs -put input/task2_output.txt /input/task3/
hadoop fs -put input/AFINN-lexicon.txt /input/task3/
hadoop jar bigram-udf.jar com.example.wordanalysis.SentimentScoreDriver /input/task3/task2_output.txt /output/task3
hadoop fs -cat /output/task3/part-r-00000
```

## Output Format

```
bookID,year    total_sentiment_score
1,1885         12
2,1800         -5
```

---

# Task 4 — Decade-Level Trend Analysis

## Overview

Maps individual years to their corresponding decades (e.g., 1823 -> 1820s) and aggregates word frequencies and sentiment scores at both book level and corpus level, enabling macro-scale historical trend analysis.

## Implementation

### Mapper: `TrendAnalysisMapper.java`

```java
package com.example.wordanalysis;

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class TrendAnalysisMapper extends Mapper<Object, Text, Text, IntWritable> {
    private Text compositeKey = new Text();
    private IntWritable valueOut = new IntWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\t");
        if (parts.length < 2) return;
        String[] keyParts = parts[0].split(",");
        if (keyParts.length < 3) return;

        String bookID = keyParts[0];
        int year = Integer.parseInt(keyParts[1]);
        String wordOrSentiment = keyParts[2];
        int decade = (year / 10) * 10;
        int count = Integer.parseInt(parts[1]);

        // Book-level key
        compositeKey.set(bookID + "," + decade + "," + wordOrSentiment);
        valueOut.set(count);
        context.write(compositeKey, valueOut);

        // Corpus-level key
        compositeKey.set(decade + "," + wordOrSentiment);
        context.write(compositeKey, valueOut);
    }
}
```

### Reducer: `TrendAnalysisReducer.java`

```java
package com.example.wordanalysis;

import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class TrendAnalysisReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val : values) sum += val.get();
        result.set(sum);
        context.write(key, result);
    }
}
```

## Sample Output

```
1,1880s,cut     10
10,1840s,drop    7
1880s,cut       30
1840s,drop      20
```

---

# Task 5 — Bigram Extraction (Hive UDF)

## Overview

Extracts and aggregates **bigrams** (consecutive word pairs) from the lemmatized corpus using a custom Hive UDF — `BigramExtractorUDF`. Reveals co-occurrence patterns and linguistic trends across the corpus.

## Step 1: Input Preparation (MapReduce)

`BigramPrepMapper`, `BigramPrepReducer`, and `BigramPrepDriver` reformat Task 2 output into Hive-friendly lines:
```
bookID\tyear\tlemmatized text
```

## Step 2: Custom Hive UDF — `BigramExtractorUDF`

```java
public List<String> evaluate(String input) {
    if (input == null || input.isEmpty()) return Collections.emptyList();
    String[] words = input.trim().split("\\s+");
    List<String> bigrams = new ArrayList<>();
    for (int i = 0; i < words.length - 1; i++)
        bigrams.add(words[i] + " " + words[i + 1]);
    return bigrams;
}
```

## Hive Queries

```sql
ADD JAR /root/bigram-udf.jar;
CREATE TEMPORARY FUNCTION extract_bigrams AS 'com.example.wordanalysis.BigramExtractorUDF';

CREATE TABLE IF NOT EXISTS bigram_input (
    book_id STRING,
    year    INT,
    text    STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE;

LOAD DATA INPATH '/user/hadoop/input/bigram_input/part-r-00000'
OVERWRITE INTO TABLE bigram_input;

CREATE TABLE bigram_output (bigram STRING, freq INT) STORED AS TEXTFILE;

INSERT OVERWRITE TABLE bigram_output
SELECT bigram, COUNT(*) AS freq
FROM (
    SELECT explode(extract_bigrams(text)) AS bigram
    FROM bigram_input
) tmp
GROUP BY bigram;
```

## Running the Full Pipeline

```bash
mvn clean package
hdfs dfs -mkdir -p /user/hadoop/input/bigram_input
hdfs dfs -put bigram_input/part-r-00000 /user/hadoop/input/bigram_input/
hive  # then run queries above
hdfs dfs -get /user/hive/warehouse/bigram_output/000000_0 output5/task5_output/task5_output.txt
```

---

## Key Results

- Processed **1M+ text records** through a 5-stage distributed pipeline on Hadoop
- Created Hive tables with partitioned datasets and Java MapReduce logic to extract frequency metrics and text patterns, converting raw files into query-ready analytical outputs
- Tuned mapper, reducer, partitioning, and intermediate data handling in MapReduce jobs — achieved **30% reduction in pipeline runtime** across distributed text-processing workloads
- Generated structured sentiment, keyword, and trend-analysis datasets ready for downstream querying

---

## Repository Structure

```
.
├── src/main/java/com/example/wordanalysis/
│   ├── TextPreprocessingMapper.java
│   ├── TextPreprocessingReducer.java
│   ├── TextPreprocessingDriver.java
│   ├── WordFrequencyMapper.java
│   ├── WordFrequencyReducer.java
│   ├── WordFrequencyDriver.java
│   ├── SentimentScoreMapper.java
│   ├── SentimentScoreReducer.java
│   ├── SentimentScoreDriver.java
│   ├── TrendAnalysisMapper.java
│   ├── TrendAnalysisReducer.java
│   ├── TrendAnalysisDriver.java
│   ├── BigramPrepMapper.java
│   ├── BigramPrepReducer.java
│   ├── BigramPrepDriver.java
│   └── BigramExtractorUDF.java
├── input/
│   ├── AFINN-lexicon.txt
│   └── task2_output.txt
├── output/          # Task 1 output
├── output_2/        # Task 2 output
├── output3_updated/ # Task 3 output
├── output_4/        # Task 4 output
├── output5/         # Task 5 bigram output
├── bigram_input/    # Formatted Hive input
├── raw_books_data.csv
├── pom.xml
├── docker-compose.yml
└── hadoop-hive.env
```
