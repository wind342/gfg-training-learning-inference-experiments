import fs from "node:fs";
import { SourceMapConsumer, SourceMapGenerator } from "source-map";

const [stage1Path, stage2Path, baseUrl1, baseUrl2, outputMapPath, outputReportPath] = process.argv.slice(2);
if (!outputReportPath) throw new Error("usage: native_map_composition.mjs M1 M2 BASE1 BASE2 OUT_MAP OUT_REPORT");
const map1 = JSON.parse(fs.readFileSync(stage1Path, "utf8"));
const map2 = JSON.parse(fs.readFileSync(stage2Path, "utf8"));
const first = await new SourceMapConsumer(map1, baseUrl1);
const second = await new SourceMapConsumer(map2, baseUrl2);
const generator = new SourceMapGenerator({ file: map2.file });
const records = [];
let broken = 0;
second.eachMapping((tail) => {
  const generated = { line: tail.generatedLine, column: tail.generatedColumn };
  if (tail.source == null) {
    generator.addMapping({ generated });
    records.push({ generated_file: map2.file, generated_line: tail.generatedLine - 1, generated_column: tail.generatedColumn, mapped: false, original_source: null, original_line: null, original_column: null, original_name: null });
    return;
  }
  const head = first.originalPositionFor({ line: tail.originalLine, column: tail.originalColumn, bias: SourceMapConsumer.GREATEST_LOWER_BOUND });
  if (head.source == null) {
    broken += 1;
    return;
  }
  generator.addMapping({
    generated,
    source: head.source,
    original: { line: head.line, column: head.column },
    name: head.name ?? tail.name ?? undefined,
  });
  const content = first.sourceContentFor(head.source, true);
  if (content != null) generator.setSourceContent(head.source, content);
  records.push({
    generated_file: map2.file,
    generated_line: tail.generatedLine - 1,
    generated_column: tail.generatedColumn,
    mapped: true,
    original_source: head.source,
    original_line: head.line - 1,
    original_column: head.column,
    original_name: head.name ?? tail.name ?? null,
  });
}, null, SourceMapConsumer.GENERATED_ORDER);
records.sort((a, b) => a.generated_line - b.generated_line || a.generated_column - b.generated_column);
fs.writeFileSync(outputMapPath, `${generator.toString()}\n`, "utf8");
fs.writeFileSync(outputReportPath, `${JSON.stringify({ records, broken_bridge_count: broken })}\n`, "utf8");
first.destroy();
second.destroy();
