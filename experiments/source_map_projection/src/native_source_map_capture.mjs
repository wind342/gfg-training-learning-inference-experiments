import fs from "node:fs";
import { SourceMapGenerator } from "source-map";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) throw new Error("usage: native_source_map_capture.mjs INPUT OUTPUT");
const payload = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const receipts = payload.receipts
  .filter((row) => row.receipt_type === "emit")
  .sort((left, right) =>
    left.generated_start.line - right.generated_start.line ||
    left.generated_start.column - right.generated_start.column
  );
const anchors = new Set();
const generator = new SourceMapGenerator({ file: payload.generated_file, sourceRoot: payload.source_root ?? undefined });
for (const receipt of receipts) {
  const anchorKey = `${receipt.generated_start.line}:${receipt.generated_start.column}`;
  if (anchors.has(anchorKey)) throw new Error(`DUPLICATE_GENERATED_ANCHOR:${anchorKey}`);
  anchors.add(anchorKey);
  const generated = { line: receipt.generated_start.line + 1, column: receipt.generated_start.column };
  if (!receipt.mapping_eligible) {
    generator.addMapping({ generated });
    continue;
  }
  const candidates = receipt.origins.filter((origin) => origin.mapping_anchor);
  if (candidates.length !== 1) throw new Error(`CONFLICTING_ORIGINAL_MAPPING:${anchorKey}`);
  const origin = candidates[0];
  generator.addMapping({
    generated,
    source: origin.source_file,
    original: { line: origin.source_start.line + 1, column: origin.source_start.column },
    name: origin.original_name ?? undefined,
  });
  generator.setSourceContent(origin.source_file, origin.source_content);
}
fs.writeFileSync(outputPath, `${generator.toString()}\n`, "utf8");
