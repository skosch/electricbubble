import React from "react";
import * as opentypejs from "opentype.js";
import * as tf from "@tensorflow/tfjs";
import update from "immutability-helper";
import unpack from "ndarray-unpack";

import {imshowAsDataUrl, tfshowAsDataUrl} from "./imshow";
import {orientedDOG, orientedGaussian, electricBubbleKernel} from "./kernels";
import {findBestDistance} from "./autofit";
import "./style.scss";


export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      sampleText: "hamburg",
      font: null,
      scalingFactor: 0.03,
      fontSize: 72,
      glyphData: {},
      parameters: {
        inkSaliency: {
          filterSize: 20,
          patchSizeX: 75,
          patchSizeY: 12,
          valleyPeakRatio: 6.2,
          valleyGain: 0.5,
          angle: 0,
          gain: 0.26,
          bias: 0.76,
        },
        whitespaceSaliency: {
          filterSize: 20,
          patchSizeX: 0.01,
          patchSizeY: 27,
          angle: 1.5708,
          gain: 0.2,
        },
        electricBubble: {
          filterSize: 30,
          power: 22,
          decay: 0.07,
        },
      },
      kernels: {
        inkSaliency: null,
        whitespaceSaliency: null,
        electricBubble: null,
      },
      kernelImages: {
        inkSaliency: null,
        whitespaceSaliency: null,
        electricBubble: null,
      },
      filteredGlyphImages: {
        inkSaliency: {},
        whitespaceSaliency: {},
        electricBubble: {},
      },
      filteredGlyphTensors: {
        inkSaliency: {},
        whitespaceSaliency: {},
        electricBubble: {},
      }
    };
    this.canvasRef = React.createRef();
    this.canvasEl = null;
    this.ctx = null;
  }

  componentDidMount() {
    this.canvasEl = this.canvasRef.current;
    this.ctx = this.canvasEl.getContext("2d");
  }

  loadFontFile = (files) => {
    const bufferFr = new FileReader();
    bufferFr.onload = (e) => {
      const font = opentypejs.parse(e.target.result);
      const scalingFactor = this.state.fontSize / font.unitsPerEm;
      const ascender = Math.round(font.ascender * scalingFactor);
      const descender = Math.round(-font.descender * scalingFactor);
      const padWidth = Math.round(ascender / 2);
      const boxHeight = ascender + descender;

      this.setState({font, scalingFactor, ascender, descender, padWidth, boxHeight});
      this.renderRawGlyphs();
    };

    bufferFr.readAsArrayBuffer(files[0]);
  }

  renderRawGlyphs = () => {
    const glyphData = {};
    for (const g of "abcdefghijklmnopqrstuvwxyz") {
      const d = this.renderRawGlyph(g);
      glyphData[g] = d;
    }
    this.setState({glyphData});
  }

  updateInkSaliencyKernel = () => tf.tidy(() => {
    const kernel = orientedDOG(this.state.parameters.inkSaliency);
    const kernelImageDataUrl = imshowAsDataUrl(kernel, this.canvasEl, this.ctx, {colormap: "picnic", min: -1, max: 1});
    const kernelTensor = tf.transpose(tf.tensor2d(unpack(kernel.selection)));

    if (this.state.kernels.inkSaliency) {
      this.state.kernels.inkSaliency.dispose(); // clear out old kernel;
    }
    this.setState(update(this.state, {
      kernels: {inkSaliency: {$set: kernelTensor}},
      kernelImages: {inkSaliency: {$set: kernelImageDataUrl}}
    }), () => this.refilterGlyphs("ink"));

    return kernelTensor;
  })

  updateWhitespaceSaliencyKernel = () => tf.tidy(() => {
    const kernel = orientedGaussian(this.state.parameters.whitespaceSaliency);
    const kernelImageDataUrl = imshowAsDataUrl(kernel, this.canvasEl, this.ctx, {colormap: "picnic", min: -1, max: 1});
    const kernelTensor = tf.transpose(tf.tensor2d(unpack(kernel.selection)));

    if (this.state.kernels.whitespaceSaliency) {
      this.state.kernels.whitespaceSaliency.dispose(); // clear out old kernel;
    }
    this.setState(update(this.state, {
      kernels: {whitespaceSaliency: {$set: kernelTensor}},
      kernelImages: {whitespaceSaliency: {$set: kernelImageDataUrl}}
    }), () => this.refilterGlyphs("whitespace"));

    return kernelTensor;
  })

  updateElectricBubbleKernel = () => tf.tidy(() => {
    const kernel = electricBubbleKernel(this.state.parameters.electricBubble);
    const kernelImageDataUrl = imshowAsDataUrl(kernel, this.canvasEl, this.ctx, {colormap: "picnic", min: 0, max: 1});
    const kernelTensor = tf.transpose(tf.tensor2d(unpack(kernel.selection)))

    if (this.state.kernels.electricBubble) {
      this.state.kernels.electricBubble.dispose(); // clear out old kernel;
    }
    this.setState(update(this.state, {
      kernels: {electricBubble: {$set: kernelTensor}},
      kernelImages: {electricBubble: {$set: kernelImageDataUrl}}
    }), () => this.refilterGlyphs("electricBubble"));

    return kernelTensor;
  })

  refilterInkSaliency = (cb) => {
    const isK = tf.expandDims(tf.expandDims(this.state.kernels.inkSaliency, -1), -1);
    const isBias = tf.scalar(this.state.parameters.inkSaliency.bias);
    const inkSaliencyTensors = {};
    const inkSaliencyImages = {};

    // dispose of old tensors
    for (let g in this.state.filteredGlyphTensors) {
      if (this.state.filteredGlyphTensors.inkSaliency[g]) {
        this.state.filteredGlyphTensors.inkSaliency[g].dispose();
      }
    }

    for (let g of this.state.sampleText) {
      const isc = tf.relu(tf.mul(tf.add(isBias, tf.conv2d(this.state.glyphData[g].rawTensor, isK, [1, 1], 'same')), this.state.glyphData[g].rawTensor));
      inkSaliencyTensors[g] = isc;
      inkSaliencyImages[g] = tfshowAsDataUrl(tf.squeeze(isc), this.canvasEl, this.ctx, {colormap: "hot", min: 3, max: 0});
    }
    isK.dispose(); isBias.dispose();

    this.setState(update(this.state, {filteredGlyphTensors: {inkSaliency: {$set: inkSaliencyTensors}},
                                      filteredGlyphImages: {inkSaliency: {$set: inkSaliencyImages}}}), cb);
  }

  refilterWhitespaceSaliency = (cb) => {
    const wsK = tf.expandDims(tf.expandDims(this.state.kernels.whitespaceSaliency, -1), -1);
    const tfOne = tf.scalar(1.0);
    const whitespaceSaliencyTensors = {};
    const whitespaceSaliencyImages = {};

    // dispose of old tensors
    for (let g in this.state.filteredGlyphTensors) {
      if (this.state.filteredGlyphTensors.whitespaceSaliency[g]) {
        this.state.filteredGlyphTensors.whitespaceSaliency[g].dispose();
      }
    }

    for (let g of this.state.sampleText) {
      const wsc = tf.conv2d(this.state.glyphData[g].rawTensor, wsK, [1, 1], 'same'); //, tf.sub(tfOne, this.state.glyphData[g].rawTensor));
      whitespaceSaliencyTensors[g] = wsc;
      whitespaceSaliencyImages[g] = tfshowAsDataUrl(tf.squeeze(wsc), this.canvasEl, this.ctx, {colormap: "hot", min: 3, max: 0});
    }

    tfOne.dispose(); wsK.dispose();

    this.setState(update(this.state, {filteredGlyphTensors: {whitespaceSaliency: {$set: whitespaceSaliencyTensors}},
                                      filteredGlyphImages: {whitespaceSaliency: {$set: whitespaceSaliencyImages}}}), cb);
  }

  refilterElectricBubble = (cb) => {
    const ebK = tf.expandDims(tf.expandDims(this.state.kernels.electricBubble, -1), -1);
    const electricBubbleImages = {};
    const electricBubbleTensors = {};
    const minClip = tf.scalar(Number.MIN_VALUE);
    const maxClip = tf.scalar(50.);
    const decay = tf.scalar(-this.state.parameters.electricBubble.decay);

    // dispose of old tensors
    for (let g in this.state.filteredGlyphTensors) {
      if (this.state.filteredGlyphTensors.electricBubble[g]) {
        this.state.filteredGlyphTensors.electricBubble[g].dispose();
      }
    }

    for (let g of this.state.sampleText) {
      const totalSaliency = tf.maximum(this.state.filteredGlyphTensors.inkSaliency[g], this.state.filteredGlyphTensors.whitespaceSaliency[g]);
      const totalSaliencyPower = tf.pow(totalSaliency, this.state.parameters.electricBubble.power);
      const wsc = tf.maximum(minClip, tf.conv2d(totalSaliencyPower, ebK, [1, 1], 'same')); // at least 0.000..001
      const ebc = tf.squeeze(tf.exp(tf.mul(decay, tf.minimum(maxClip, tf.pow(wsc, -1./this.state.parameters.electricBubble.power)))));
      electricBubbleTensors[g] = ebc;
      electricBubbleImages[g] = tfshowAsDataUrl(ebc /* already squeezed */, this.canvasEl, this.ctx, {colormap: "hot", min: 1.0, max: 0});
      totalSaliency.dispose(); totalSaliencyPower.dispose(); wsc.dispose();
    }

    minClip.dispose(); maxClip.dispose(); decay.dispose();

    this.setState(update(this.state, {filteredGlyphTensors: {electricBubble: {$set: electricBubbleTensors}},
                                      filteredGlyphImages: {electricBubble: {$set: electricBubbleImages}}}), cb);
  }

  refilterGlyphs = (updatedParamType) => {
    if (updatedParamType === "ink") {
      this.refilterInkSaliency(() => {
        if (this.state.kernels.electricBubble) {
          this.refilterElectricBubble();
        }
      });
    }
    if (updatedParamType === "whitespace") {
      this.refilterWhitespaceSaliency(() => {
        if (this.state.kernels.electricBubble) {
          this.refilterElectricBubble();
        }
      });
    }
    if (updatedParamType === "electricBubble") {
      this.refilterElectricBubble();
    }
  }

  autofit = () => {
    for (let i = 0; i < this.state.sampleText.length - 1; i++) {
      const g1 = this.state.sampleText[i];
      const g2 = this.state.sampleText[i + 1];
      const bestoverlap = findBestDistance(this.state.filteredGlyphTensors.electricBubble[g1], this.state.filteredGlyphTensors.electricBubble[g2], this.state);
      console.log(g1, g2, bestoverlap);
    }
  }

  updateParam = (updater, kernelFunc) => {
    this.setState(update(this.state, {parameters: updater}), kernelFunc);
  }

  renderRawGlyph = (g) => tf.tidy(() => {
    const glyph = this.state.font.stringToGlyphs(g)[0];
    const {xMin, xMax} = glyph.getMetrics();
    const glyphLeft = Math.round(xMin * this.state.scalingFactor);
    const glyphWidth = Math.floor((xMax - xMin) * this.state.scalingFactor);
    // resize the canvas and fill it with white
    this.canvasEl.height = this.state.boxHeight;
    this.canvasEl.width = glyphWidth + 2;
    this.ctx.fillStyle = "white";
    this.ctx.fillRect(0, 0, this.canvasEl.width, this.canvasEl.height);
    // draw the glyph
    glyph.draw(this.ctx, 1 - glyphLeft, this.state.ascender, this.state.fontSize);
    const imageData = this.ctx.getImageData(0, 0, 1 + glyphWidth, this.state.boxHeight);
    let rawTensorCropped = tf.scalar(255).sub(tf.slice(tf.browser.fromPixels(this.canvasEl, 3), [0, 0, 0], [-1, -1, 1])).div(tf.scalar(255));
    const rawTensor = tf.pad(tf.expandDims(rawTensorCropped, 0), [[0, 0], [0, 0], [this.state.padWidth, this.state.padWidth], [0, 0]])
    const imageDataUrl = this.canvasEl.toDataURL();

    return {rawTensorCropped, rawTensor, imageDataUrl};
  });

  render() {
    return (
      <div className="columns">
        <div className="column is-one-quarter">
          <h1>Electric bubble</h1>
          <input type="file" onChange={e => this.loadFontFile(e.target.files)} />
          <div className="parameter-panel" style={{display: this.state.font ? "block" : "none"}}>
          <div>
            <h3>Ink saliency filter</h3>
            <label>Filter size ({this.state.parameters.inkSaliency.filterSize})
              <input type="range" min="3" max="300" step={2}
                     value={this.state.parameters.inkSaliency.filterSize}
                     onChange={(e) => this.updateParam({inkSaliency: {filterSize: {$set: parseInt(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Sigma X ({this.state.parameters.inkSaliency.patchSizeX})
              <input type="range" min="0.01" max="100" step={0.01}
                     value={this.state.parameters.inkSaliency.patchSizeX}
                     onChange={(e) => this.updateParam({inkSaliency: {patchSizeX: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Sigma Y ({this.state.parameters.inkSaliency.patchSizeY})
              <input type="range" min="0.01" max="100" step={0.01}
                     value={this.state.parameters.inkSaliency.patchSizeY}
                     onChange={(e) => this.updateParam({inkSaliency: {patchSizeY: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Valley size (relative) ({this.state.parameters.inkSaliency.valleyPeakRatio})
              <input type="range" min="0.1" max="10" step={0.1}
                     value={this.state.parameters.inkSaliency.valleyPeakRatio}
                     onChange={(e) => this.updateParam({inkSaliency: {valleyPeakRatio: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Valley depth (relative) ({this.state.parameters.inkSaliency.valleyGain})
              <input type="range" min="0.01" max="1.0" step={0.01}
                     value={this.state.parameters.inkSaliency.valleyGain}
                     onChange={(e) => this.updateParam({inkSaliency: {valleyGain: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Bias ({this.state.parameters.inkSaliency.bias})
              <input type="range" min="0.0" max="3.0" step={0.01}
                     value={this.state.parameters.inkSaliency.bias}
                     onChange={(e) => this.updateParam({inkSaliency: {bias: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Gain ({this.state.parameters.inkSaliency.gain})
              <input type="range" min="0.1" max="3.0" step={0.01}
                     value={this.state.parameters.inkSaliency.gain}
                     onChange={(e) => this.updateParam({inkSaliency: {gain: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <label>Angle ({this.state.parameters.inkSaliency.angle})
              <input type="range" min="-1.578" max="1.578" step={0.001}
                     value={this.state.parameters.inkSaliency.angle}
                     onChange={(e) => this.updateParam({inkSaliency: {angle: {$set: parseFloat(e.target.value)}}}, this.updateInkSaliencyKernel)} />
            </label>
            <img src={this.state.kernelImages.inkSaliency} />
          </div>
          <div>
            <h3>Whitespace saliency filter</h3>
            <label>Filter size ({this.state.parameters.whitespaceSaliency.filterSize})
              <input type="range" min="3" max="300" step={2}
                     value={this.state.parameters.whitespaceSaliency.filterSize}
                     onChange={(e) => this.updateParam({whitespaceSaliency: {filterSize: {$set: parseInt(e.target.value)}}}, this.updateWhitespaceSaliencyKernel)} />
            </label>
            <label>Sigma X ({this.state.parameters.whitespaceSaliency.patchSizeX})
              <input type="range" min="0.001" max="0.2" step={0.001}
                     value={this.state.parameters.whitespaceSaliency.patchSizeX}
                     onChange={(e) => this.updateParam({whitespaceSaliency: {patchSizeX: {$set: parseFloat(e.target.value)}}}, this.updateWhitespaceSaliencyKernel)} />
            </label>
            <label>Sigma Y ({this.state.parameters.whitespaceSaliency.patchSizeY})
              <input type="range" min="0.01" max="100" step={0.01}
                     value={this.state.parameters.whitespaceSaliency.patchSizeY}
                     onChange={(e) => this.updateParam({whitespaceSaliency: {patchSizeY: {$set: parseFloat(e.target.value)}}}, this.updateWhitespaceSaliencyKernel)} />
            </label>
            <label>Gain ({this.state.parameters.whitespaceSaliency.gain})
              <input type="range" min="0.01" max="1.0" step={0.01}
                     value={this.state.parameters.whitespaceSaliency.gain}
                     onChange={(e) => this.updateParam({whitespaceSaliency: {gain: {$set: parseFloat(e.target.value)}}}, this.updateWhitespaceSaliencyKernel)} />
            </label>
            <label>Angle ({this.state.parameters.whitespaceSaliency.angle})
              <input type="range" min="-1.578" max="1.578" step={0.001}
                     value={this.state.parameters.whitespaceSaliency.angle}
                     onChange={(e) => this.updateParam({whitespaceSaliency: {angle: {$set: parseFloat(e.target.value)}}}, this.updateWhitespaceSaliencyKernel)} />
            </label>
            <img src={this.state.kernelImages.whitespaceSaliency} />
          </div>
          <div>
            <h3>Electric bubble</h3>
            <label>Filter size ({this.state.parameters.electricBubble.filterSize})
              <input type="range" min="3" max="100" step={2}
                     value={this.state.parameters.electricBubble.filterSize}
                     onChange={(e) => this.updateParam({electricBubble: {filterSize: {$set: parseInt(e.target.value)}}}, this.updateElectricBubbleKernel)} />
            </label>
            <label>Power ({this.state.parameters.electricBubble.power})
              <input type="range" min="1" max="25" step={1}
                     value={this.state.parameters.electricBubble.power}
                     onChange={(e) => this.updateParam({electricBubble: {power: {$set: parseFloat(e.target.value)}}}, this.updateElectricBubbleKernel)} />
            </label>
            <label>Decay ({this.state.parameters.electricBubble.decay})
              <input type="range" min="0.001" max="1.0" step={0.001}
                     value={this.state.parameters.electricBubble.decay}
                     onChange={(e) => this.updateParam({electricBubble: {decay: {$set: parseFloat(e.target.value)}}}, this.updateElectricBubbleKernel)} />
            </label>
            <img src={this.state.kernelImages.electricBubble} />
          </div>
          <button onClick={this.autofit}>Autofit</button>
        </div>
        </div>
        <div className="column">
          <input onChange={e => this.setState({sampleText: e.target.value})} value={this.state.sampleText} />
          <div>
          {String.split(this.state.sampleText, "").map((g, gi) => (
            (this.state.glyphData[g] ? (
              <img src={this.state.glyphData[g].imageDataUrl || ""} key={gi} />
            ) : null)
          ))}
          </div>
          <div>
          {String.split(this.state.sampleText, "").map((g, gi) => (
            <img src={this.state.filteredGlyphImages.inkSaliency[g] || ""} key={gi} />
          ))}
          </div>
          <div>
          {String.split(this.state.sampleText, "").map((g, gi) => (
            <img src={this.state.filteredGlyphImages.whitespaceSaliency[g] || ""} key={gi} />
          ))}
          </div>
          <div>
          {String.split(this.state.sampleText, "").map((g, gi) => (
            <img src={this.state.filteredGlyphImages.electricBubble[g] || ""} key={gi} />
          ))}
          </div>
        </div>
        <canvas ref={this.canvasRef} id="render-canvas" width={600} height={400} style={{display: "none"}}></canvas>
      </div>
    );
  }
}
