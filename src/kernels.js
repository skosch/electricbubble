import nj from "numjs";
import ndarray from "ndarray";

const rotationMatrix = (angle) => {
  return nj.array([[Math.cos(angle), -Math.sin(angle)], [Math.sin(angle), Math.cos(angle)]]);
};

const unscaledMultivariateGaussian = (filterSize, covarianceMatrix, rotM) => {
  // rotated covariance matrix
  const sigma = nj.dot(rotM, nj.dot(covarianceMatrix, rotM.T));
  const detSigma = sigma.get(0, 0) * sigma.get(1, 1) - sigma.get(0, 1) * sigma.get(1, 0);
  const invSigma = nj.array([[sigma.get(1, 1), -sigma.get(0, 1)], [-sigma.get(1, 0), sigma.get(0, 0)]]).divide(detSigma); /// detSigma;
  const [s_a, s_b, s_c, s_d] = [invSigma.get(0, 0), invSigma.get(0, 1), invSigma.get(1, 0), invSigma.get(1, 1)];

  const result = nj.zeros([filterSize, filterSize]);
  const hfs = filterSize / 2;
  for (let y = 0; y < filterSize; y++) {
    for (let x = 0; x < filterSize; x++) {
      const dx = x - hfs;
      const dy = y - hfs;
      const z = Math.exp(-(dx*(dx*s_a + dy*s_c) + dy*(dx*s_b + dy*s_d)));
      result.set(y, x, z);
    }
  }

  return result;
};

/* Oriented Difference of Gaussians (for the whitespace blurring)
/* Parameters: {filterSize, patchSizeX, patchSizeY, angle, gain} */
export const orientedGaussian = (params) => {
  const rotM = rotationMatrix(params.angle);
  const peak = unscaledMultivariateGaussian(params.filterSize, nj.array([[params.patchSizeX, 0], [0, params.patchSizeY]]), rotM);
  return peak.multiply(params.gain);
};

/* Oriented Difference of Gaussians (instead of a Gabor kernel)
/* Parameters: {filterSize, patchSizeX, patchSizeY, valleyPeakRatio, angle, gain} */
export const orientedDOG = (params) => {
  const rotM = rotationMatrix(params.angle);
  const peak = unscaledMultivariateGaussian(params.filterSize, nj.array([[params.patchSizeX, 0], [0, params.patchSizeY]]), rotM);
  const valley = unscaledMultivariateGaussian(params.filterSize, nj.array([[params.patchSizeX * params.valleyPeakRatio, 0], [0, params.patchSizeY * params.valleyPeakRatio]]), rotM);
  const envelope = unscaledMultivariateGaussian(params.filterSize, nj.array([[params.patchSizeX / 2, 0], [0, 1000 * params.filterSize]]), rotM);
  return peak.subtract(valley.multiply(params.valleyGain)).multiply(envelope).multiply(params.gain);
};

/* Electric potential / approximate distance transform kernel */
export const electricBubbleKernel = (params) => {
  const result = nj.zeros([params.filterSize, params.filterSize]);
  const hfs = Math.round(params.filterSize / 2);
  const halfpower = params.power / 2; // instead of sqrt-ing and then squaring again
  for (let y = 0; y < params.filterSize; y++) {
    for (let x = 0; x < params.filterSize; x++) {
      const dx = x - hfs;
      const dy = y - hfs;
      const dist = (dx * dx + dy * dy) || 1; // no sqrt here, but we'll only use half the power
      const z = 1 / (dist ** halfpower);
      result.set(y, x, z);
    }
  }
  return result;
};
