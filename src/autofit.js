import * as tf from "@tensorflow/tfjs";
import {imshowAsDataUrl, tfshowAsDataUrl} from "./imshow";

export const findBestDistance = (bubble1, dx1, dy1, bubble2, dx2, dy2, state, canvasEl, ctx) => tf.tidy(() => {
  // we want to find the first distance at which the bubbles interact too much.

  const unfilteredBorder = Math.round(Math.max(1, state.padWidth - state.parameters.electricBubble.filterSize / 2 - 2));
  let desirableInteraction = 0.6;
  let meanInteraction = 0;
  // instead of the max, we want the angle-weighted sum to be something.
  let overlap = Math.round(Math.min(Math.max(1, unfilteredBorder + 5) + 30, bubble1.shape[1] * 0.8, bubble2.shape[1] * 0.8));
  let maxAttempts = 80;
  let significance = null;
  let weightedInteraction = null;
  let totalRequiredInteraction = null;

  let equidistFactor = null;
  let angleDiffFactor = null;
  let meanAngleFactor = null;
  let attention = null;
  let product = null;

  let yFactors = state.parameters.yFactors.split(",").map(f => parseFloat(f.trim()));

  for (let attempts = 0; attempts < maxAttempts && overlap < bubble1.shape[1] * 0.9 && overlap < bubble1.shape[1] * 0.9 && overlap >= 1 && overlap < 2 * state.padWidth; attempts++) {
    const rightBubbleSlice = tf.slice(bubble1, [0, bubble1.shape[1] - overlap], [-1, -1]);
    const leftBubbleSlice = tf.slice(bubble2, [0, 0], [-1, overlap]);
    const rightDxSlice = tf.slice(dx1, [0, dx1.shape[1] - overlap], [-1, -1]);
    const rightDySlice = tf.slice(dy1, [0, dy1.shape[1] - overlap], [-1, -1]);
    const leftDxSlice = tf.slice(dx2, [0, 0], [-1, overlap]);
    const leftDySlice = tf.slice(dy2, [0, 0], [-1, overlap]);

    const sq = a => tf.pow(a, tf.scalar(2.0));
    const avg = (a, b) => tf.div(tf.add(a, b), tf.scalar(2.0));

    // equidistFactor, normalized by the average magnitude of the two bubbles. Works pretty well.
    equidistFactor = tf.exp(tf.mul(tf.scalar(-state.parameters.equidistImportance),
      tf.div(tf.squaredDifference(leftBubbleSlice, rightBubbleSlice), sq(avg(leftBubbleSlice, rightBubbleSlice)))));

    // angle diff, measures how straight it is. -1 for same direction, 1 for completely straight.
    // function: sigmoid function that limits output to straight angles
    const angleDiffLength = tf.sqrt(tf.add(tf.mul(tf.add(sq(rightDxSlice), sq(rightDySlice)), tf.add(sq(leftDxSlice), sq(leftDySlice))), tf.scalar(1.e-21)));
    const angleDiffCosine = tf.div( tf.mul(-1., tf.add(tf.mul(leftDySlice, rightDySlice), tf.mul(rightDxSlice, leftDxSlice))) , angleDiffLength);
    angleDiffFactor = tf.sigmoid(tf.mul(tf.sub(angleDiffCosine, 1.0), tf.scalar(state.parameters.angleDifferenceCoefficient)));
    //angleDiffFactor = tf.relu(tf.add(tf.mul(angleDiffCosine, tf.scalar(1.0+state.parameters.angleDifferenceCoefficient)), tf.scalar(-state.parameters.angleDifferenceCoefficient)));

    // mean angle factor, which should
    const ty = tf.sub(rightDySlice, leftDySlice)
    const tx = tf.sub(rightDxSlice, leftDxSlice)
    const meanAngleLength = tf.sqrt(tf.add(tf.add(sq(tx), sq(ty)), tf.scalar(1.e-21)))
    const meanAngleSine = tf.div( tf.abs(ty) , meanAngleLength );
    meanAngleFactor = tf.add(tf.scalar(1.0), tf.mul(tf.scalar(state.parameters.meanAngleCoefficient), meanAngleSine));

    attention = tf.mul(tf.mul(equidistFactor, angleDiffFactor), meanAngleFactor);
    product = tf.mul(attention, tf.pow(tf.add(rightBubbleSlice, leftBubbleSlice), tf.scalar(1.0)));

    totalRequiredInteraction = tf.scalar(state.parameters.desiredInteraction);

    //const meanAngle = tf.add(tf.abs(rightAngleSlice), tf.abs(leftAngleSlice));
//    significance = tf.mul(attention, tf.exp(tf.mul(tf.scalar(-state.parameters.angleDifferenceCoefficient), angleDifference)));
    //const meanAngleDifference = tf.div(tf.mul(angleDifference, product), tf.sum(product)).dataSync()[0, 0];
 //   weightedInteraction = tf.div(tf.mul(product, significance), tf.sum(significance)); // only consider the parallelism where the interaction is strong.

    //const maxInteraction = tf.max(tf.mul(rightBubbleSlice, leftBubbleSlice)).dataSync()[0, 0];
    //desirableInteraction = 0.5 + state.parameters.angleDifferenceCoefficient * meanAngleDifference;

    meanInteraction = tf.div(tf.sum(product), tf.sum(attention)).dataSync()[0, 0];
    if (Math.abs(meanInteraction - totalRequiredInteraction.dataSync()[0, 0]) < 0.001 || attempts === maxAttempts - 1) {
      console.log("Mean interaction was:", meanInteraction, "at overlap", overlap, "but desired was", totalRequiredInteraction.dataSync()[0, 0], "after attempts:", attempts);
      //return (2 * state.padWidth - 2) - overlap;
      break;
    } else {
      //console.log("mean interaction was", meanInteraction, "at overlap", overlap);
      const grad = meanInteraction - totalRequiredInteraction.dataSync()[0, 0];
      overlap -= (Math.round(grad) || (grad > 0 ? 1 : -1));
      if (attempts === maxAttempts - 1 || overlap >= bubble1.shape[1] * 0.9 || overlap >= bubble2.shape[1] * 0.9 || overlap < 1 || overlap >= 2 * state.padWidth) {
        console.log("Quitting while Mean interaction was:", meanInteraction, "at overlap", overlap, "but desired was", totalRequiredInteraction.dataSync()[0, 0])
      }
    }
  }

  const equidistImage = tfshowAsDataUrl(equidistFactor, canvasEl, ctx, {colormap: "picnic"});
  const angleDiffImage = tfshowAsDataUrl(angleDiffFactor, canvasEl, ctx, {colormap: "picnic"});
  const meanAngleImage = tfshowAsDataUrl(meanAngleFactor, canvasEl, ctx, {colormap: "picnic"});
  const attentionImage = tfshowAsDataUrl(attention, canvasEl, ctx, {colormap: "picnic"});
  const weightedInteractionImage = tfshowAsDataUrl(product, canvasEl, ctx, {colormap: "picnic"});

  return [(2 * state.padWidth) - overlap, weightedInteractionImage, attentionImage, equidistImage, angleDiffImage, meanAngleImage];
});
