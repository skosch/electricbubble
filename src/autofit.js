import * as tf from "@tensorflow/tfjs";

export const findBestDistance = (bubble1, bubble2, state) => tf.tidy(() => {
  // we want to find the first distance at which the bubbles interact too much.

  const unfilteredBorder = Math.max(1, state.padWidth - state.parameters.electricBubble.filterSize / 2 - 2);

  for (let overlap = unfilteredBorder; overlap < unfilteredBorder + 50; overlap++) {
    const rightSlice = tf.slice(bubble1, [0, bubble1.shape[1] - overlap], [-1, -1]);
    const leftSlice = tf.slice(bubble2, [0, 0], [-1, overlap]);
    const maxInteraction = tf.max(tf.mul(rightSlice, leftSlice)).dataSync()[0, 0];
    if (maxInteraction > 0.5) {
      return overlap;
    }
  }
  return unfilteredBorder + 50;
});
