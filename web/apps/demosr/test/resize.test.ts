/**
 * The demo's MATLAB bicubic against gaussifier_sr.matlab_resize. fixtures/resize.json holds a
 * 32x24 test image and the Python results (matlab_downsample_uint8 at 2 and 4, and the bicubic
 * baseline np.clip(np.round(imresize(lr / 255, s) * 255), 0, 255) of each). When the SR goldens
 * exist (scripts/export_web_goldens.py), the Kodak LR images are checked as well.
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { imresize8, matlabDownsample, type Rgb8 } from "../src/resize.js";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/resize.json", import.meta.url), "utf8"));
const image: Rgb8 = { data: Uint8Array.from(fixture.image), width: fixture.width, height: fixture.height };

describe("MATLAB bicubic", () => {
  for (const s of [2, 4]) {
    it(`downsamples by ${s} like matlab_downsample_uint8`, () => {
      const lr = matlabDownsample(image, s);
      expect([lr.width, lr.height]).toEqual(fixture[`down${s}_size`]);
      expect(Array.from(lr.data)).toEqual(fixture[`down${s}`]);
    });
    it(`upsamples by ${s} like the evaluation's bicubic baseline`, () => {
      const lr: Rgb8 = { data: Uint8Array.from(fixture[`down${s}`]), width: fixture[`down${s}_size`][0], height: fixture[`down${s}_size`][1] };
      const up = imresize8(lr, s);
      expect([up.width, up.height]).toEqual(fixture[`up${s}_size`]);
      expect(Array.from(up.data)).toEqual(fixture[`up${s}`]);
    });
  }
});

const goldens = new URL("../../../fixtures/sr/", import.meta.url);
const cases = existsSync(goldens) ? readdirSync(goldens).filter((d) => existsSync(new URL(`${d}/lr_u8.bin`, goldens))) : [];
describe.skipIf(cases.length === 0)("SR goldens' LR images", () => {
  for (const name of cases) {
    it(`${name}: the LR image matches`, () => {
      const m = JSON.parse(readFileSync(new URL(`${name}/manifest.json`, goldens), "utf8"));
      const hr: Rgb8 = { data: new Uint8Array(readFileSync(new URL(`${name}/hr_u8.bin`, goldens))), width: m.hr.width, height: m.hr.height };
      const lr = matlabDownsample(hr, m.scale);
      const expected = new Uint8Array(readFileSync(new URL(`${name}/lr_u8.bin`, goldens)));
      let mismatches = 0;
      for (let i = 0; i < expected.length; i++) if (lr.data[i] !== expected[i]) mismatches++;
      expect(mismatches).toBe(0);
    });
  }
});
