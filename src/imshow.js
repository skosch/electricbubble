import ndarray from "ndarray";
import pool from "typedarray-pool";
import applyColormap from "apply-colormap";
import ops from "ndarray-ops";
import unpack from "ndarray-unpack";

const colorize = (img, options, cb) => {
  options = options || {}
  var gray = !!(options.grayscale ||
                options.greyscale ||
                options.luminance ||
                options.gray ||
                options.grey ||
                options.colormap === "gray")
  var buf = pool.mallocUint8(img.size * 3);
  var opts = { "outBuffer": buf }
  if ("min" in options) {
    opts.min = +options.min
  }
  if ("max" in options) {
    opts.max = +options.max
  }
  if ("colormap" in options) {
    opts.colormap = options.colormap
  }
  if (gray) {
    opts.colormap = "greys"
  }
  var result = applyColormap(img, opts)
  pool.freeUint8(buf);
  return result;
}

export const imshowAsDataUrl = (inputArray, canvasEl, ctx, options) => {
  // if numjs array (which as .selection), get the actual ndarray first. Then run the colorize
  let array = null;
  if (options.colormap !== "none") {
    array = colorize(inputArray.hasOwnProperty("selection") ? inputArray.selection : inputArray, options);
  } else {
    array = inputArray.hasOwnProperty("selection") ? inputArray.selection : inputArray;
  }

  canvasEl.height = array.shape[0];
  canvasEl.width = array.shape[1];

  // imageData is empty data array in the current shape of the canvas
  let imageData = ctx.getImageData(0, 0, canvasEl.width, canvasEl.height);

  if (array.shape.length === 3) {
    // height, width, 3 color channels
    ops.assign(ndarray(imageData.data, [array.shape[0], array.shape[1], 3], [4, 4 * array.shape[0], 1]), array)
    ops.assigns(ndarray(imageData.data, [array.shape[0] * array.shape[1]], [4], 3), 255)
  } else if (array.shape.length === 2) {
    // height, width
    ops.assign(ndarray(imageData.data, [array.shape[0], array.shape[1], 3], [4, 4 * array.shape[0], 1]), ndarray(array.data, [array.shape[0], array.shape[1], 3], [array.stride[0], array.stride[1], 0], array.offset));
    ops.assigns(ndarray(imageData.data, [array.shape[0] * array.shape[1]], [4], 3), 255);
  }

  ctx.putImageData(imageData, 0, 0);

  return canvasEl.toDataURL();
}

export const tfshowAsDataUrl = (tensor, canvasEl, ctx, options) => {
  const array = ndarray(tensor.dataSync(), [tensor.shape[0], tensor.shape[1]], [1, tensor.shape[0]]);
  return imshowAsDataUrl(array, canvasEl, ctx, options);
}
