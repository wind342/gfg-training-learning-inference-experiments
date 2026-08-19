import fs from "node:fs";
import { SourceMapConsumer } from "source-map";

function normalizeOriginal(row) {
  return {
    source: row.source ?? null,
    line: row.line == null ? null : row.line - 1,
    column: row.column ?? null,
    name: row.name ?? null,
  };
}

async function decode(mapPath, baseUrl) {
  const raw = JSON.parse(fs.readFileSync(mapPath, "utf8"));
  const consumer = await new SourceMapConsumer(raw, baseUrl);
  const records = [];
  consumer.eachMapping((mapping) => {
    records.push({
      generated_file: raw.file ?? null,
      generated_line: mapping.generatedLine - 1,
      generated_column: mapping.generatedColumn,
      mapped: mapping.source != null,
      original_source: mapping.source ?? null,
      original_line: mapping.originalLine == null ? null : mapping.originalLine - 1,
      original_column: mapping.originalColumn ?? null,
      original_name: mapping.name ?? null,
    });
  }, null, SourceMapConsumer.GENERATED_ORDER);
  consumer.destroy();
  return { records, sources: raw.sources ?? [], names: raw.names ?? [], sourcesContent: raw.sourcesContent ?? null };
}

async function query(mapPath, baseUrl, cohortPath) {
  const raw = JSON.parse(fs.readFileSync(mapPath, "utf8"));
  const cohort = JSON.parse(fs.readFileSync(cohortPath, "utf8"));
  const consumer = await new SourceMapConsumer(raw, baseUrl);
  const results = [];
  for (const item of cohort) {
    try {
      if (item.direction === "generated_to_original") {
        const answer = consumer.originalPositionFor({
          line: item.generated_line + 1,
          column: item.generated_column,
          bias: item.bias === "LUB" ? SourceMapConsumer.LEAST_UPPER_BOUND : SourceMapConsumer.GREATEST_LOWER_BOUND,
        });
        results.push({ query_id: item.query_id, direction: item.direction, status: "ok", answers: [normalizeOriginal(answer)] });
      } else if (item.direction === "original_to_generated") {
        const answers = consumer.allGeneratedPositionsFor({
          source: item.original_source,
          line: item.original_line + 1,
          column: item.original_column,
        }).map((answer) => ({ line: answer.line - 1, column: answer.column, last_column: answer.lastColumn ?? null }));
        answers.sort((a, b) => a.line - b.line || a.column - b.column);
        results.push({ query_id: item.query_id, direction: item.direction, status: "ok", answers });
      } else {
        throw new Error("QUERY_DIRECTION_UNKNOWN");
      }
    } catch (error) {
      results.push({ query_id: item.query_id, direction: item.direction, status: "controlled_error", reason: error.name });
    }
  }
  consumer.destroy();
  return results;
}

const [command, mapPath, baseUrl, arg4, arg5] = process.argv.slice(2);
if (command === "decode") {
  fs.writeFileSync(arg4, `${JSON.stringify(await decode(mapPath, baseUrl))}\n`, "utf8");
} else if (command === "query") {
  fs.writeFileSync(arg5, `${JSON.stringify(await query(mapPath, baseUrl, arg4))}\n`, "utf8");
} else {
  throw new Error("usage: source_map_consumer_oracle.mjs decode MAP BASE OUT | query MAP BASE COHORT OUT");
}
