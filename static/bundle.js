(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
const BASE_URL = 'http://127.0.0.1:5001';

async function updateWeightsWithBackend(weights, inputs, targets) {
    const response = await fetch(`${BASE_URL}/update-weights`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ weights, inputs, targets }),
    });

    if (!response.ok) {
        throw new Error('Failed to update weights');
    }

    const data = await response.json();
    return data.updated_weights;
}

// Export for CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { updateWeightsWithBackend };
}

// Attach to window for browser
if (typeof window !== 'undefined') {
    window.updateWeightsWithBackend = updateWeightsWithBackend;
}

},{}],2:[function(require,module,exports){
/*globals paper, console, $ */
/*jslint nomen: true, undef: true, sloppy: true */

// kmedoids library

/*

@licstart  The following is the entire license notice for the
JavaScript code in this page.

Copyright (C) 2015 david ha, otoro.net, otoro labs

The JavaScript code in this page is free software: you can
redistribute it and/or modify it under the terms of the GNU
General Public License (GNU GPL) as published by the Free Software
Foundation, either version 3 of the License, or (at your option)
any later version.  The code is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU GPL for more details.

As additional permission under GNU GPL version 3 section 7, you
may distribute non-source (e.g., minimized or compacted) forms of
that code without the copy of the GNU GPL normally required by
section 4, provided you include this license notice and a URL
through which recipients can access the Corresponding Source.


@licend  The above is the entire license notice
for the JavaScript code in this page.
*/

// implementation of neat algorithm with recurrent.js graphs to support backprop
// used for genetic art.
// code is not modular or oop, ad done in a fortran/scientific computing style
// apologies in advance if that ain't ur taste.
if (typeof module != "undefined") {
  var R = require('./recurrent.js');
  //var NetArt = require('./netart.js');
}

var kmedoids = {};

(function(global) {
  "use strict";

  // Utility fun
  var assert = function(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  };

  var cluster =[];
  var nCluster = 1;
  var LARGENUM = 10e20;

  var initCluster = function() {
    var i;
    cluster = [];
    for (i=0;i<nCluster;i++) {
      cluster.push([]);
    }
  };

  var init = function(nCluster_) {
    nCluster = nCluster_;
    initCluster();
  };

  var shuffle = function(origArray) {
    // returns a newArray which is a shuffled version of origArray
    var i, randomIndex;
    var temp;
    var N = origArray.length;
    var result = new Array(N);
    for (i=0;i<N;i++) {
      result[i] = origArray[i];
    }
    for (i=0;i<N;i++) {
      // swaps i with randomIndex
      randomIndex = R.randi(0, N);
      temp = result[randomIndex];
      result[randomIndex] = result[i];
      result[i] = temp;
    }
    return result;
  };

  var copyArray = function(origArray) {
    // returns a newArray which is a copy of origArray
    var i;
    var N = origArray.length;
    var result = new Array(N);
    for (i=0;i<N;i++) {
      result[i] = origArray[i];
    }
    return result;
  };

  var equalArray = function(array1, array2) {
    // returns true if two arrays are equal
    if (array1.length !== array2.length) return false;
    var i;
    var N = array1.length;
    for (i=0;i<N;i++) {
      if (array1[i] !== array2[i]) return false;
    }
    return true;
  };

  var swapArray = function(array, idx1, idx2) {
    // swap array[idx1] and array[idx2]
    var temp = array[idx1];
    array[idx1] = array[idx2];
    array[idx2] = temp;
  };

  var dist = function(e1, e2) { // function can be overloaded to other dist func
    var dx = e2.x-e1.x;
    var dy = e2.y-e1.y;
    return (dx*dx+dy*dy); //removed sqr root
  };

  function clusterLength() {
    var i, j, m, n;
    var count=0;
    for (i=0,m=cluster.length;i<m;i++) {
      for (j=0,n=cluster[i].length;j<n;j++) {
        count++;
      }
    }
    return count;
  }

  var lloyd_partition = function(list) {
    var i, j, m, n, idx, d, d2, k, cost, bestCost, anchor;

    n = list.length;
    var distTable = new R.Mat(n, n);

    for (i=0;i<n;i++) {
      for (j=0;j<=i;j++) {
        if (i===j) {
          distTable.set(i, j, 0);
        }
        d = dist(list[i], list[j]);
        distTable.set(i, j, d);
        distTable.set(j, i, d);
      }
    }

    var idxArray = new Array(n);
    for (i=0;i<n;i++) {
      idxArray[i] = i;
    }
    idxArray = shuffle(idxArray);

    var anchorArray = idxArray.splice(0, nCluster);

    var oldAnchorArray;

    var maxTry = 100;
    var countTry = 0;

    do {

      oldAnchorArray = copyArray(anchorArray);
      initCluster(); // wipes out cluster

      // push anchor array into clusters first as the first element
      for (j=0;j<nCluster;j++) {
        cluster[j].push(anchorArray[j]);
      }

      // go thru remaining idx Arrays to assign each element to closest anchor
      for (i=0,n=idxArray.length;i<n;i++) {
        k=-1;
        d=LARGENUM;
        for (j=0;j<nCluster;j++) {
          d2 = distTable.get(idxArray[i], anchorArray[j]);
          if (d2 < d) {
            k = j;
            d = d2;
          }
        }
        assert(k>=0, 'cannot find closest distance to index; all distances greater than '+LARGENUM);
        cluster[k].push(idxArray[i]);
      }

      // for each cluster, reassign the anchor position
      for (i=0;i<nCluster;i++) {
        anchor=-1;
        bestCost=LARGENUM;
        n = cluster[i].length;
        for (j=0;j<n;j++) {
          cost = 0;
          for (k=0;k<n;k++) {
            cost += distTable.get(cluster[i][j],cluster[i][k]);
          }
          if (cost < bestCost) {
            bestCost = cost;
            anchor = j;
          }
        }
        assert(anchor>=0, 'cannot find a good anchor position');
        swapArray(cluster[i], 0, anchor);
      }

      // reprocess the clusters back into array and repeat until it converges
      idxArray = [];
      for (i=0;i<nCluster;i++) {
        anchorArray[i] = cluster[i][0];
        n = cluster[i].length;
        for (j=1;j<n;j++) {
          idxArray.push(cluster[i][j]);
        }
      }

      countTry++;

      if (countTry >= maxTry) {
        // console.log('k-medoids did not converge after '+maxTry+ ' tries.');
        break;
      }

    } while(!equalArray(anchorArray, oldAnchorArray));

  };

  var pam_partition = function(list) {
    var i, j, m, n, idx, d, d2, k, cost, bestCost, anchor, temp;

    bestCost = LARGENUM;

    n = list.length;
    var distTable = new R.Mat(n, n);

    for (i=0;i<n;i++) {
      for (j=0;j<=i;j++) {
        if (i===j) {
          distTable.set(i, j, 0);
        }
        d = dist(list[i], list[j]);
        distTable.set(i, j, d);
        distTable.set(j, i, d);
      }
    }

    var idxArray = new Array(n);
    for (i=0;i<n;i++) {
      idxArray[i] = i;
    }
    idxArray = shuffle(idxArray);

    var anchorArray = idxArray.splice(0, nCluster);

    var oldAnchorArray;

    var maxTry = 100;
    var countTry = 0;

    function buildCluster() {
      var i, j, k, n, d, d2;
      var localCost = 0;
      initCluster(); // wipes out cluster

      // push anchor array into clusters first as the first element
      for (j=0;j<nCluster;j++) {
        cluster[j].push(anchorArray[j]);
      }

      // go thru remaining idx Arrays to assign each element to closest anchor
      for (i=0,n=idxArray.length;i<n;i++) {
        k=-1;
        d=LARGENUM;
        for (j=0;j<nCluster;j++) {
          d2 = distTable.get(idxArray[i], anchorArray[j]);
          if (d2 < d) {
            k = j;
            d = d2;
          }
        }
        assert(k>=0, 'cannot build cluster since all distances from anchor is greater than '+LARGENUM);
        cluster[k].push(idxArray[i]);
        localCost += d;
      }
      return localCost;
    }

    do {

      oldAnchorArray = copyArray(anchorArray);

      bestCost = buildCluster();

      for (i=0;i<nCluster;i++) {
        for (j=0,n=idxArray.length;j<n;j++) {
          // swap
          temp = anchorArray[i];
          anchorArray[i] = idxArray[j];
          idxArray[j] = temp;

          cost = buildCluster();

          // swap back if it doesn't work
          if (cost > bestCost) {
            temp = anchorArray[i];
            anchorArray[i] = idxArray[j];
            idxArray[j] = temp;
          } else {
            bestCost = cost;
          }
        }
      }

      bestCost = buildCluster();
      //console.log('best cost = '+bestCost);

      countTry++;

      if (countTry >= maxTry) {
        // console.log('k-medoids did not converge after '+maxTry+ ' tries.');
        break;
      }

    } while(!equalArray(anchorArray, oldAnchorArray));

  };

  var partition = pam_partition;

  var getCluster = function() {
    return cluster;
  };

  var pushToCluster = function(elementIdx, idx_) {
    var idx = 0;
    if (typeof idx_ !== 'undefined') idx = idx_;
    cluster[idx].push(elementIdx);
  };

  global.init = init;
  global.setDistFunction = function(distFunc) {
    dist = distFunc;
  };
  global.getCluster = getCluster;
  global.partition = partition;
  global.pushToCluster = pushToCluster;

})(kmedoids);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(kmedoids);



},{"./recurrent.js":4}],3:[function(require,module,exports){
/*globals paper, console, $ */
/*jslint nomen: true, undef: true, sloppy: true */
// NEAT implementation
/*

@licstart  The following is the entire license notice for the
JavaScript code in this page.

Copyright (C) 2016 david ha, otoro.net, otoro labs

MIT License.

@licend  The above is the entire license notice
for the JavaScript code in this page.
*/

// implementation of neat algorithm with recurrent.js graphs
// code is not modular or oop, ad done in a fortran/scientific computing style
// apologies in advance if that ain't ur taste.
if (typeof module != "undefined") {
  var R = require('./recurrent.js');
  var kmedoids = require('./kmedoids.js');
}

var N = {};

(function(global) {
  "use strict";

  var updateWeightsWithBackend;

  if (typeof module !== 'undefined' && module.exports) {
      // Node.js/CommonJS
      const api = require('./api.js');
      updateWeightsWithBackend = api.updateWeightsWithBackend;
  } else if (typeof window !== 'undefined') {
      // Browser
      updateWeightsWithBackend = window.updateWeightsWithBackend;
  } else {
      throw new Error('Environment not supported: updateWeightsWithBackend not found');
  }


  // constants
  var NODE_INPUT = 0;
  var NODE_OUTPUT = 1;
  var NODE_BIAS = 2;
  // hidden layers
  var NODE_SIGMOID = 3;
  var NODE_TANH = 4;
  var NODE_RELU = 5;
  var NODE_GAUSSIAN = 6;
  var NODE_SIN = 7;
  var NODE_COS = 8;
  var NODE_ABS = 9;
  var NODE_MULT = 10;
  var NODE_ADD = 11;
  var NODE_MGAUSSIAN = 12; // multi-dim gaussian (do gaussian to each input then mult)
  var NODE_SQUARE = 13;

  var NODE_INIT = NODE_ADD;

  var MAX_TICK = 100;

  var operators = [null, null, null, 'sigmoid', 'tanh', 'relu', 'gaussian', 'sin', 'cos', 'abs', 'mult', 'add', 'mult', 'add'];

  // for connections
  var IDX_CONNECTION = 0;
  var IDX_WEIGHT = 1;
  var IDX_ACTIVE = 2;

  var debug_mode = false; // shuts up the messages.

  // for initialisation ("none" means initially just 1 random connection, "one" means 1 node that connects all, "all" means fully connected)
  var initConfig = "none"; // default for generating images.

  //var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_MGAUSSIAN, NODE_SQUARE];
  //var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_ADD];
  // var activations = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_SQUARE];

  // keep below for generating images, for the default.
  var activations_default = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_ABS, NODE_MULT, NODE_SQUARE, NODE_ADD];
  var activations_all = [NODE_SIGMOID, NODE_TANH, NODE_RELU, NODE_GAUSSIAN, NODE_SIN, NODE_COS, NODE_MULT, NODE_ABS, NODE_ADD, NODE_MGAUSSIAN, NODE_SQUARE];
  var activations_minimal = [NODE_RELU, NODE_TANH, NODE_GAUSSIAN, NODE_ADD];

  var activations = activations_default;

  var getRandomActivation = function() {
    var ix = R.randi(0, activations.length);
    return activations[ix];
  };

  var gid = 0;
  var getGID = function() {
    var result = gid;
    gid += 1;
    return result;
  };

  var nodes = []; // this array holds all nodes
  var connections = []; // index of connections here is the 'innovation' value

  var copyArray = function(x) {
    // returns a copy of floatArray
    var n = x.length;
    var result = new Array(n);
    for (var i=0;i<n;i++) {
      result[i]=x[i];
    }
    return result;
  };

  function copyConnections(newC) {
    var i, n;
    n = newC.length;
    var copyC = [];
    for (i=0;i<n;i++) { // connects input and bias to init dummy node
      copyC.push([newC[i][0], newC[i][1]]);
    }
    return copyC;
  }

  var getNodes = function() {
    return copyArray(nodes);
  };

  var getConnections = function() {
    return copyConnections(connections);
  };

  var nInput = 1;
  var nOutput = 1;
  var outputIndex = 2; // [bias, input, output]
  var initNodes = [];
  var initMu = 0.0, initStdev = 1.0; // randomised param initialisation.
  var mutationRate = 0.2;
  var mutationSize = 0.5;
  var generationNum = 0;

  var incrementGenerationCounter = function() { generationNum += 1; };

  function getRandomRenderMode() {
    // more chance to be 0 (1/2), and then 1/3 to be 1, and 1/6 to be 2
    var z = R.randi(0, 6);
    if (z<3) return 0;
    if (z<5) return 1;
    return 2;
  }

  var renderMode = getRandomRenderMode(); // 0 = sigmoid (1 = gaussian, 2 = tanh+abs

  var randomizeRenderMode = function() {
    renderMode = getRandomRenderMode();
    if (debug_mode) console.log('render mode = '+renderMode);
  };

  var setRenderMode = function(rMode) {
    renderMode = rMode;
  };

  var getRenderMode = function() {
    return renderMode;
  };

  function getOption(opt, index, orig) {
    if (opt && typeof opt[index] !== null) { return opt[index]; }
    return orig;
  }

  var init = function(opt) {
    var i, j;
    nInput = getOption(opt, 'nInput', nInput);
    nOutput = getOption(opt, 'nOutput', nOutput);
    initConfig = getOption(opt, 'initConfig', initConfig);
    if (typeof opt.activations !== "undefined" && opt.activations === "all") {
      activations = activations_all;
    }
    if (typeof opt.activations !== "undefined" && opt.activations === "minimal") {
      activations = activations_minimal;
      //operators[NODE_OUTPUT] = 'tanh'; // for minimal, make output be an tanh operator.
    }

    outputIndex = nInput+1; // index of output start (bias so add 1)
    nodes = [];
    connections = [];
    generationNum = 0; // initialisze the gen counter
    // initialise nodes
    for (i=0;i<nInput;i++) {
      nodes.push(NODE_INPUT);
    }
    nodes.push(NODE_BIAS);
    for (i=0;i<nOutput;i++) {
      nodes.push(NODE_OUTPUT);
    }
    // initialise connections. at beginning only connect inputs to outputs
    // initially, bias has no connections and that must be grown.

    if (initConfig === "all") {
      for (j=0;j<nOutput;j++) {
        for (i=0;i<nInput+1;i++) {
          connections.push([i, outputIndex+j]);
        }
      }
    } else if (initConfig === "one") {
      // push initial dummy node
      nodes.push(NODE_ADD);
      var dummyIndex = nodes.length-1;
      for (i=0;i<nInput+1;i++) { // connects input and bias to init dummy node
        connections.push([i, dummyIndex]);
      }
      for (i=0;i<nOutput;i++) { // connects dummy node to output
        connections.push([dummyIndex, outputIndex+i]);
      }
    }


  };

  function getNodeList(node_type) {
    // returns a list of locations (index of global node array) containing
    // where all the output nodes are
    var i, n;
    var result = [];
    for (i=0,n=nodes.length;i<n;i++) {
      if (nodes[i] === node_type) {
        result.push(i);
      }
    }
    return result;
  }

  var Genome = function(initGenome) {
    var i, j;
    var n;
    var c; // connection storage.

    this.connections = [];
    // create or copy initial connections
    if (initGenome && typeof initGenome.connections !== null) {
      for (i=0,n=initGenome.connections.length;i<n;i++) {
        this.connections.push(R.copy(initGenome.connections[i]));
      }
    } else {

      if (initConfig === "all") {
        // copy over initial connections (nInput + connectBias) * nOutput
        for (i=0,n=(nInput+1)*nOutput;i<n;i++) {
          c = R.zeros(3); // innovation number, weight, enabled (1)
          c[IDX_CONNECTION] = i;
          c[IDX_WEIGHT] = R.randn(initMu, initStdev);
          c[IDX_ACTIVE] = 1;
          this.connections.push(c);
        }
      }  else if (initConfig === "one") {

        for (i=0,n=(nInput+1)+nOutput;i<n;i++) {
          c = R.zeros(3); // innovation number, weight, enabled (1)
          c[IDX_CONNECTION] = i;
          // the below line assigns 1 to initial weights from dummy node to output
          c[IDX_WEIGHT] = (i < (nInput+1)) ? R.randn(initMu, initStdev) : 1.0;
          //c[IDX_WEIGHT] = R.randn(initMu, initStdev);
          c[IDX_ACTIVE] = 1;
          this.connections.push(c);
        }

      }

    }
  };

  Genome.prototype = {
    copy: function() {
      // makes a copy of itself and return it (returns a Genome class)
      var result = new Genome(this);
      // copies other tags if exist
      if (this.fitness) result.fitness = this.fitness;
      if (this.cluster) result.cluster = this.cluster;
      return result;
    },
    importConnections: function(cArray) {
      var i, n;
      this.connections = [];
      var temp;
      for (i=0,n=cArray.length;i<n;i++) {
        temp = new R.zeros(3);
        temp[0] = cArray[i][0];
        temp[1] = cArray[i][1];
        temp[2] = cArray[i][2];
        this.connections.push(temp);
      }
    },
    copyFrom: function(sourceGenome) {
      // copies connection weights from sourceGenome to this genome, hence making a copy
      this.importConnections(sourceGenome.connections);
      if (sourceGenome.fitness) this.fitness = sourceGenome.fitness;
      if (sourceGenome.cluster) this.cluster = sourceGenome.cluster;
    },
    mutateWeights: async function(mutationRate_, mutationSize_) {
      // mutates each weight of current genome with a probability of mutationRate
      // by adding a gaussian noise of zero mean and mutationSize stdev to it
      var mRate = mutationRate_ || mutationRate;
      var mSize = mutationSize_ || mutationSize;

      // var i, n;
      // for (i=0,n=this.connections.length;i<n;i++) {
      //   if (Math.random() < mRate) {
      //     this.connections[i][IDX_WEIGHT] += R.randn(0, mSize);
      //   }
      // }
      try {
          const updatedWeights = await global.updateWeightsWithBackend(
              this.connections.map(c => c[1]), // Export weights
              [[1.0, 0.5], [0.3, 0.7]],        // Example inputs
              [0.6, 0.8]                      // Example targets
          );

          // Update genome's weights with the response
          this.connections.forEach((c, i) => {
              c[1] = updatedWeights[i]; // Update the weight in the connection
          });
      } catch (error) {
          console.error('Error during weight optimization:', error);
      }
    },
    areWeightsNaN: function() {
      // if any weight value is NaN, then returns true and break.
      var origWeight;
      var i, n;
      for (i=0,n=this.connections.length;i<n;i++) {
          origWeight = this.connections[i][IDX_WEIGHT];
          if (isNaN(origWeight)) {
            return true;
          }
      }
      return false;
    },
    clipWeights: function(maxWeight_) {
      // weight clipping to +/- maxWeight_
      // this function also checks if weights are NaN. if so, zero them out.
      var maxWeight = maxWeight_ || 50.0;
      maxWeight = Math.abs(maxWeight);
      var origWeight;

      var detectedNaN = false;

      var i, n;
      for (i=0,n=this.connections.length;i<n;i++) {
          origWeight = this.connections[i][IDX_WEIGHT];

          R.assert(!isNaN(origWeight), 'weight had NaN.  backprop messed shit up.');

          origWeight = Math.min(maxWeight, origWeight);
          origWeight = Math.max(-maxWeight,origWeight);

          this.connections[i][IDX_WEIGHT] = origWeight;
      }
    },
    exportWeights: function() {
      const weights = [];
      const nConnections = this.model.connections.length;
      for (let i = 0; i < nConnections; i++) {
        weights.push(this.model.connections[i].w[0]);
      }
      return weights;
    },
    importWeights : function(weightArray) {
      const nConnections = this.model.connections.length;
      if (weightArray.length !== nConnections) {
        throw new Error("Weight array length does not match the number of connections!");
      }
      for (let i = 0; i < nConnections; i++) {
        this.model.connections[i].w[0] = weightArray[i];
      }
    },
    getAllConnections: function() {
      return connections;
    },
    addRandomNode: function() {
      // adds a new random node and assigns it a random activation gate
      // if there are no connections, don't add a new node
      if (this.connections.length === 0) return;
      var c = R.randi(0, this.connections.length); // choose random connection
      // only proceed if the connection is actually active.
      if (this.connections[c][IDX_ACTIVE] !== 1) return;

      var w = this.connections[c][1];

      this.connections[c][IDX_ACTIVE] = 0; // disable the connection
      var nodeIndex = nodes.length;
      nodes.push(getRandomActivation()); // create the new node globally

      var innovationNum = this.connections[c][0];
      var fromNodeIndex = connections[innovationNum][0];
      var toNodeIndex = connections[innovationNum][1];

      var connectionIndex = connections.length;
      // make 2 new connection globally
      connections.push([fromNodeIndex, nodeIndex]);
      connections.push([nodeIndex, toNodeIndex]);

      // put in this node locally into genome
      var c1 = R.zeros(3);
      c1[IDX_CONNECTION] = connectionIndex;
      c1[IDX_WEIGHT] = 1.0; // use 1.0 as first connection weight
      c1[IDX_ACTIVE] = 1;
      var c2 = R.zeros(3);
      c2[IDX_CONNECTION] = connectionIndex+1;
      c2[IDX_WEIGHT] = w; // use old weight for 2nd connection
      c2[IDX_ACTIVE] = 1;

      this.connections.push(c1);
      this.connections.push(c2);
    },
    getNodesInUse: function() {
      var i, n, connectionIndex, nodeIndex;
      var nodesInUseFlag = R.zeros(nodes.length);
      var nodesInUse = [];
      var len = nodes.length;

      for (i=0,n=this.connections.length;i<n;i++) {
        connectionIndex = this.connections[i][0];
        nodeIndex = connections[connectionIndex][0];
        nodesInUseFlag[nodeIndex] = 1;
        nodeIndex = connections[connectionIndex][1];
        nodesInUseFlag[nodeIndex] = 1;
      }
      for (i=0,n=len;i<n;i++) {
        if (nodesInUseFlag[i] === 1 || (i < nInput+1+nOutput) ) { // if node is input, bias, output, throw it in too
          //console.log('pushing node #'+i+' as node in use');
          nodesInUse.push(i);
        }
      }
      return nodesInUse;
    },
    addRandomConnection: function() {
      // attempts to add a random connection.
      // if connection exists, then does nothing (ah well)

      var i, n, connectionIndex, nodeIndex;

      var nodesInUse = this.getNodesInUse();
      var len = nodes.length;

      //var fromNodeIndex = R.randi(0, nodes.length);
      //var toNodeIndex = R.randi(outputIndex, nodes.length); // includes bias.

      var slack = 0;
      var r1 = R.randi(0, nodesInUse.length - nOutput);
      if (r1 > nInput+1) slack = nOutput; // skip the outputs of the array.
      var fromNodeIndex = nodesInUse[r1 + slack]; // choose anything but output nodes
      var toNodeIndex = nodesInUse[R.randi(outputIndex, nodesInUse.length)]; // from output to other nodes

      var fromNodeUsed = false;
      var toNodeUsed = false;

      if (fromNodeIndex === toNodeIndex) {
        //console.log('addRandomConnection failed to connect '+fromNodeIndex+' to '+toNodeIndex);
        return; // can't be the same index.
      }

      // cannot loop back out from the output.
      /*
      if (fromNodeIndex >= outputIndex && fromNodeIndex < (outputIndex+nOutput)){
        //console.log('addRandomConnection failed to connect '+fromNodeIndex+' to '+toNodeIndex);
        return;
      }
      */

      // the below set of code will test if selected nodes are actually used in network connections
      for (i=0,n=this.connections.length;i<n;i++) {
        connectionIndex = this.connections[i][0];
        if ((connections[connectionIndex][0] === fromNodeIndex) || (connections[connectionIndex][1] === fromNodeIndex)) {
          fromNodeUsed = true; break;
        }
      }
      for (i=0,n=this.connections.length;i<n;i++) {
        connectionIndex = this.connections[i][0];
        if ((connections[connectionIndex][0] === toNodeIndex) || (connections[connectionIndex][1] === toNodeIndex)) {
          toNodeUsed = true; break;
        }
      }

      if (fromNodeIndex < nInput+1) fromNodeUsed = true; // input or bias
      if ((toNodeIndex >= outputIndex) && (toNodeIndex < outputIndex+nOutput)) toNodeUsed = true; // output

      if (!fromNodeUsed || !toNodeUsed) {
        if (debug_mode) {
          console.log('nodesInUse.length = '+nodesInUse.length);
          console.log('addRandomConnection failed to connect '+fromNodeIndex+' to '+toNodeIndex);
        }
        return; // only consider connections in current net.
      }
      //console.log('attempting to connect '+fromNodeIndex+' to '+toNodeIndex);

      var searchIndex = -1; // see if connection already exist.
      for (i=0,n=connections.length;i<n;i++) {
        if (connections[i][0] === fromNodeIndex && connections[i][1] === toNodeIndex) {
          searchIndex = i; break;
        }
      }

      if (searchIndex < 0) {
        // great, this connection doesn't exist yet!
        connectionIndex = connections.length;
        connections.push([fromNodeIndex, toNodeIndex]);

        var c = R.zeros(3); // innovation number, weight, enabled (1)
        c[IDX_CONNECTION] = connectionIndex;
        c[IDX_WEIGHT] = R.randn(initMu, initStdev);
        c[IDX_ACTIVE] = 1;
        this.connections.push(c);
      } else {
        var connectionIsInGenome = false;
        for (i=0,n=this.connections.length; i<n; i++) {
          if (this.connections[i][IDX_CONNECTION] === searchIndex) {
            // enable back the index (if not enabled)
            if (this.connections[i][IDX_ACTIVE] === 0) {
              this.connections[i][IDX_WEIGHT] = R.randn(initMu, initStdev); // assign a random weight to reactivated connections
              this.connections[i][IDX_ACTIVE] = 1;
            }
            connectionIsInGenome = true;
            break;
          }
        }
        if (!connectionIsInGenome) {
          // even though connection exists globally, it isn't in this gene.
          //console.log('even though connection exists globally, it isnt in this gene.');
          var c1 = R.zeros(3); // innovation number, weight, enabled (1)
          c1[IDX_CONNECTION] = searchIndex;
          c1[IDX_WEIGHT] = R.randn(initMu, initStdev);
          c1[IDX_ACTIVE] = 1;
          this.connections.push(c1);

          //console.log('added connection that exists somewhere else but not here.');
        }
      }

    },
    createUnrolledConnections: function() {
      // create a large array that is the size of Genome.connections
      // element:
      // 0: 1 or 0, whether this connection exists in this genome or not
      // 1: weight
      // 2: active? (1 or 0)
      var i, n, m, cIndex, c;
      this.unrolledConnections = [];
      n=connections.length; // global connection length
      m=this.connections.length;
      for (i=0;i<n;i++) {
        this.unrolledConnections.push(R.zeros(3));
      }
      for (i=0;i<m;i++) {
        c = this.connections[i];
        cIndex = c[IDX_CONNECTION];
        this.unrolledConnections[cIndex][IDX_CONNECTION] = 1;
        this.unrolledConnections[cIndex][IDX_WEIGHT] = c[IDX_WEIGHT];
        this.unrolledConnections[cIndex][IDX_ACTIVE] = c[IDX_ACTIVE];
      }
    },
    crossover: function(that) { // input is another genome
      // returns a newly create genome that is the offspring.
      var i, n, c;
      var child = new Genome();
      child.connections = []; // empty initial connections
      var g;
      var count;

      n = connections.length;

      this.createUnrolledConnections();
      that.createUnrolledConnections();

      for (i=0;i<n;i++) {
        count = 0;
        g = this;
        if (this.unrolledConnections[i][IDX_CONNECTION] === 1) {
          count++;
        }
        if (that.unrolledConnections[i][IDX_CONNECTION] === 1) {
          g = that;
          count++;
        }
        if (count === 2 && Math.random() < 0.5) {
          g = this;
        }
        if (count === 0) continue; // both genome doesn't contain this connection
        c = R.zeros(3);
        c[IDX_CONNECTION] = i;
        c[IDX_WEIGHT] = g.unrolledConnections[i][IDX_WEIGHT];
        // in the following line, the connection is disabled only of it is disabled on both parents
        c[IDX_ACTIVE] = 1;
        if (this.unrolledConnections[i][IDX_ACTIVE] === 0 && that.unrolledConnections[i][IDX_ACTIVE] === 0) {
          c[IDX_ACTIVE] = 0;
        }
        child.connections.push(c);
      }

      return child;
    },
    setupModel: function(inputDepth) {
      // setup recurrent.js model
      var i;
      var nNodes = nodes.length;
      var nConnections = connections.length;
      this.createUnrolledConnections();
      this.model = [];
      var nodeModel = [];
      var connectionModel = [];
      var c;
      for (i=0;i<nNodes;i++) {
        nodeModel.push(new R.Mat(inputDepth, 1));
      }
      for (i=0;i<nConnections;i++) {
        c = new R.Mat(1, 1);
        c.w[0] = this.unrolledConnections[i][IDX_WEIGHT];
        connectionModel.push(c);
      }
      this.model.nodes = nodeModel;
      this.model.connections = connectionModel;
    },
    updateModelWeights: function() {
      // assume setupModel is already run. updates internal weights
      // after backprop is performed
      var i, n, m, cIndex;
      var nConnections = connections.length;

      var connectionModel = this.model.connections;
      var c;

      for (i=0;i<nConnections;i++) {
        this.unrolledConnections[i][IDX_WEIGHT] = connectionModel[i].w[0];
      }

      m=this.connections.length;
      for (i=0;i<m;i++) {
        c = this.connections[i];
        cIndex = c[IDX_CONNECTION];
        if (c[IDX_ACTIVE]) {
          c[IDX_WEIGHT] = this.unrolledConnections[cIndex][IDX_WEIGHT];
        }
      }
    },
    zeroOutNodes: function() {
      R.zeroOutModel(this.model.nodes);
    },
    setInput: function(input) {
      // input is an n x d R.mat, where n is the inputDepth, and d is number of inputs
      // for generative art, d is typically just (x, y)
      // also sets all the biases to be 1.0
      // run this function _after_ setupModel() is called!
      var i, j;
      var n = input.n;
      var d = input.d;
      var inputNodeList = getNodeList(NODE_INPUT);
      var biasNodeList = getNodeList(NODE_BIAS);
      var dBias = biasNodeList.length;

      R.assert(inputNodeList.length === d, 'inputNodeList is not the same as dimentions');
      R.assert(this.model.nodes[0].n === n, 'model nodes is not the same as dimentions');

      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          this.model.nodes[inputNodeList[j]].set(i, 0, input.get(i, j));
        }
        for (j=0;j<dBias;j++) {
          this.model.nodes[biasNodeList[j]].set(i, 0, 1.0);
        }
      }
    },
    getOutput: function() {
      // returns an array of recurrent.js Mat's representing the output
      var i;
      var outputNodeList = getNodeList(NODE_OUTPUT);
      var d = outputNodeList.length;
      var output = [];
      for (i=0;i<d;i++) {
        output.push(this.model.nodes[outputNodeList[i]]);
      }
      return output;
    },
    roundWeights: function() {
      var precision = 10000;
      for (var i=0;i<this.connections.length;i++) {
        this.connections[i][IDX_WEIGHT] = Math.round(this.connections[i][IDX_WEIGHT]*precision)/precision;
      }
    },
    toJSON: function(description) {

      var data = {
        nodes: copyArray(nodes),
        connections: copyConnections(connections),
        nInput: nInput,
        nOutput: nOutput,
        renderMode: renderMode,
        outputIndex: outputIndex,
        genome: this.connections,
        description: description
      };

      this.backup = new Genome(this);

      return JSON.stringify(data);

    },
    fromJSON: function(data_string) {
      var data = JSON.parse(data_string);
      nodes = copyArray(data.nodes);
      connections = copyConnections(data.connections);
      nInput = data.nInput;
      nOutput = data.nOutput;
      renderMode = data.renderMode || 0; // might not exist.
      outputIndex = data.outputIndex;
      this.importConnections(data.genome);

      return data.description;
    },
    forward: function(G) {
      // forward props the network from input to output.  this is where magic happens.
      // input G is a recurrent.js graph
      var outputNodeList = getNodeList(NODE_OUTPUT);
      var biasNodeList = getNodeList(NODE_BIAS);
      var inputNodeList = biasNodeList.concat(getNodeList(NODE_INPUT));

      var i, j, n;
      var nNodes = nodes.length;
      var nConnections = connections.length;
      var touched = R.zeros(nNodes);
      var prevTouched = R.zeros(nNodes);
      var nodeConnections = new Array(nNodes); // array of array of connections.

      var nodeList = [];
      var binaryNodeList = R.zeros(nNodes);

      for (i=0;i<nNodes;i++) {
        nodeConnections[i] = []; // empty array.
      }

      for (i=0;i<nConnections;i++) {
        if (this.unrolledConnections[i][IDX_ACTIVE] && this.unrolledConnections[i][IDX_CONNECTION]) {
          nodeConnections[connections[i][1]].push(i); // push index of connection to output node
          binaryNodeList[connections[i][0]] = 1;
          binaryNodeList[connections[i][1]] = 1;
        }
      }

      for (i=0;i<nNodes;i++) {
        if (binaryNodeList[i] === 1) {
          nodeList.push(i);
        }
      }

      for (i=0,n=inputNodeList.length;i<n;i++) {
        touched[inputNodeList[i]] = 1.0;
      }

      function allTouched(listOfNodes) {
        for (var i=0,n=listOfNodes.length;i<n;i++) {
          if (touched[listOfNodes[i]] !== 1) {
            return false;
          }
        }
        return true;
      }

      function noProgress(listOfNodes) {
        var idx;
        for (var i=0,n=listOfNodes.length;i<n;i++) {
          idx = listOfNodes[i];
          if (touched[idx] !== prevTouched[idx]) {
            return false;
          }
        }
        return true;
      }

      function copyTouched(listOfNodes) {
        var idx;
        for (var i=0,n=listOfNodes.length;i<n;i++) {
          idx = listOfNodes[i];
          prevTouched[idx] = touched[idx];
        }
      }

      function forwardTouch() {
        var i, j;
        var n=nNodes, m, ix; // ix is the index of the global connections.
        var theNode;

        for (i=0;i<n;i++) {
          if (touched[i] === 0) {
            theNode = nodeConnections[i];
            for (j=0,m=theNode.length;j<m;j++) {
              ix = theNode[j];
              if (touched[connections[ix][0]] === 1) {
                //console.log('node '+connections[ix][0]+' is touched, so now node '+i+' has been touched');
                touched[i] = 2; // temp touch state
                break;
              }
            }
          }
        }

        for (i=0;i<n;i++) {
          if (touched[i] === 2) touched[i] = 1;
        }

      }

      // forward tick magic
      function forwardTick(model) {
        var i, j;
        var n, m, cIndex, nIndex; // ix is the index of the global connections.
        var theNode;

        var currNode, currOperand, currConnection; // recurrent js objects
        var needOperation; // don't need operation if node is operator(node) is null or mul or add
        var nodeType;
        var finOp; // operator after all operands are weighted summed or multiplied
        var op; // either 'add' or 'eltmult'
        var out; // temp variable for storing recurrentjs state
        var cumulate; // cumulate all the outs (either addition or mult)

        n=nNodes;
        for (i=0;i<n;i++) {
          if (touched[i] === 1) { // operate on this node since it has been touched

            theNode = nodeConnections[i];
            m=theNode.length;
            // if there are no operands for this node, then don't do anything.
            if (m === 0) continue;

            nodeType = nodes[i];
            needOperation = true;
            finOp = operators[nodeType];
            if (finOp === null || finOp === 'mult' || finOp === 'add' || nodeType === NODE_MGAUSSIAN) needOperation = false;

            // usually we add weighted sum of operands, except if operator is mult
            op = 'add';
            if (finOp === 'mult') op = 'eltmul';

            // cumulate all the operands
            for (j=0;j<m;j++) {
              cIndex = theNode[j];
              nIndex = connections[cIndex][0];
              currConnection = model.connections[cIndex];
              currOperand = model.nodes[nIndex];
              out = G.mul(currOperand, currConnection);
              if (nodeType === NODE_MGAUSSIAN) { // special case:  the nasty multi gaussian
                out = G.gaussian(out);
              }
              if (j === 0) { // assign first result to cumulate
                cumulate = out;
              } else { // cumulate next result after first operand
                cumulate = G[op](cumulate, out); // op is either add or eltmul
              }
            }

            // set the recurrentjs node here
            model.nodes[i] = cumulate;
            // operate on cumulated sum or product if needed
            if (needOperation) {
              model.nodes[i] = G[finOp](model.nodes[i]);
            }

            // another special case, squaring the output
            if (nodeType === NODE_SQUARE) {
              model.nodes[i] = G.eltmul(model.nodes[i], model.nodes[i]);
            }

          }
        }


      }

      function printTouched() {
        var i;
        var result="";
        for (i=0;i<touched.length;i++) {
          result += touched[i]+" ";
        }
        console.log(result);
      }

      //printTouched();
      for (i=0;i<MAX_TICK;i++) {
        forwardTouch();
        forwardTick(this.model); // forward tick the network using graph
        //printTouched();
        /*
        if (allTouched(outputNodeList)) {
          //console.log('all outputs touched!');
          //break;
        }
        */
        if (allTouched(nodeList)) {
          //console.log('all nodes touched!');
          break;
        }
        if (noProgress(nodeList)) { // the forward tick made no difference, stuck
          //console.log('all nodes touched!');
          break;
        }
        copyTouched(nodeList);
      }

    }

  };

  var NEATCompressor = function() {
    // compresses neat, given a list of genes.
  };

  NEATCompressor.prototype = {
    buildMap: function(genes) {
      // pass in an array of all the genomes that matter, to build compression map.
      var nNode = nodes.length;
      var nConnection = connections.length;
      var nGene = genes.length;
      var connectionUsage = R.zeros(nConnection);
      var nodeUsage = R.zeros(nNode);
      var i, j, gc, c, idx, nodeIndex1, nodeIndex2;
      var newConnectionCount = 0;
      var newNodeCount = 0;
      // find out which connections are actualy used by the population of genes
      for (i=0;i<nGene;i++) {
        gc = genes[i].connections;
        for (j=0;j<gc.length;j++) {
          if (gc[j][IDX_ACTIVE] === 1) {
            idx = gc[j][IDX_CONNECTION]; // index of global connections array.
            connectionUsage[idx] = 1;
          }
        }
      }
      // from the active connections, find out which nodes are actually used
      for (i=0;i<nConnection;i++) {
        if (connectionUsage[i] === 1) {
          newConnectionCount += 1;
          nodeIndex1 = connections[i][0]; // from node
          nodeIndex2 = connections[i][1]; // to node
          nodeUsage[nodeIndex1] = 1;
          nodeUsage[nodeIndex2] = 1;
        }
      }
      // count active nodes
      for (i=0;i<nNode;i++) {
        if (nodeUsage[i] === 1) {
          newNodeCount += 1;
        }
      }
      // declare maps
      this.nodeMap = R.zeros(newNodeCount);
      this.connectionMap = R.zeros(newConnectionCount);
      this.nodeReverseMap = R.zeros(nNode);
      this.connectionReverseMap = R.zeros(nConnection);
      // calculate maps
      j = 0;
      for (i=0;i<nNode;i++) {
        if (nodeUsage[i] === 1) {
          this.nodeMap[j] = i;
          this.nodeReverseMap[i] = j;
          j += 1;
        }
      }
      j = 0;
      for (i=0;i<nConnection;i++) {
        if (connectionUsage[i] === 1) {
          this.connectionMap[j] = i;
          this.connectionReverseMap[i] = j;
          j += 1;
        }
      }

      // calculate new nodes and connections arrays but store them in compressor
      // only replace live ones when comressNEAT() is called.
      this.newNodes = [];
      this.newConnections = [];
      for (i=0;i<newNodeCount;i++) {
        this.newNodes.push(nodes[this.nodeMap[i]]);
      }
      for (i=0;i<newConnectionCount;i++) {
        c = connections[this.connectionMap[i]];
        nodeIndex1 = this.nodeReverseMap[c[0]]; // fix bug here.
        nodeIndex2 = this.nodeReverseMap[c[1]];
        this.newConnections.push([nodeIndex1, nodeIndex2]);
      }
    },
    compressNEAT: function() {
      // compresses nodes and connections global vars in neat.js
      /* ie, these:
      var nodes = []; // this array holds all nodes
      var connections = []; // index of connections here is the 'innovation' value
      */
      nodes = this.newNodes;
      connections = this.newConnections;
    },
    compressGenes: function(genes) {
      // applies the compression map to an array of genomes
      var nGene = genes.length;
      var newConnections = [];
      var gc, c, oldc, i, j, w;
      var oldConnectionIndex;

      for (i=0;i<nGene;i++) {
        gc = genes[i].connections;
        newConnections = [];
        for (j=0;j<gc.length;j++) {
          oldc = gc[j];
          if (oldc[IDX_ACTIVE] === 1) {
            oldConnectionIndex = oldc[IDX_CONNECTION];
            w = oldc[IDX_WEIGHT];
            c = R.zeros(3); // innovation number, weight, enabled (1)
            c[IDX_CONNECTION] = this.connectionReverseMap[oldConnectionIndex];
            c[IDX_WEIGHT] = w;
            c[IDX_ACTIVE] = 1;
            newConnections.push(c);
          }
        }
        genes[i].connections = newConnections;
      }
    },
  };

  var NEATTrainer = function(options_, initGenome_) {
    // implementation of a variation of NEAT training algorithm based off K-medoids.
    //
    // options:
    // num_populations : positive integer, the number of sub populations we want to preserve.
    // sub_population_size : positive integer.  Note that this is the population size for each sub population
    // hall_of_fame_size : positive integer, stores best guys in all of history and keeps them.
    // new_node_rate : [0, 1], when mutation happens, chance of a new node being added
    // new_connection_rate : [0, 1], when mutation happens, chance of a new connection being added
    // extinction_rate : [0, 1], probability that crappiest subpopulation is killed off during evolve()
    // mutation_rate : [0, 1], when mutation happens, chance of each connection weight getting mutated
    // mutation_size : positive floating point.  stdev of gausian noise added for mutations
    // init_weight_magnitude : stdev of initial random weight (default = 1.0)
    // debug_mode: false by default.  if set to true, console.log debug output occurs.
    // target_fitness : after fitness achieved is greater than this float value, learning stops
    // initGenome_: model NEAT genome to initialize with. can be result obtained from pretrained sessions.

    var options = options_ || {};

    this.num_populations = typeof options.num_populations !== 'undefined' ? options.num_populations : 5;
    this.sub_population_size = typeof options.sub_population_size !== 'undefined' ? options.sub_population_size : 10;
    this.hall_of_fame_size = typeof options.hall_of_fame_size !== 'undefined' ? options.hall_of_fame_size : 5;

    this.new_node_rate = typeof options.new_node_rate !== 'undefined' ? options.new_node_rate : 0.1;
    this.new_connection_rate = typeof options.new_connection_rate !== 'undefined' ? options.new_connection_rate : 0.1;
    this.extinction_rate = typeof options.extinction_rate !== 'undefined' ? options.extinction_rate : 0.5;
    this.mutation_rate = typeof options.mutation_rate !== 'undefined' ? options.mutation_rate : 0.1;
    this.mutation_size = typeof options.mutation_size !== 'undefined' ? options.mutation_size : 1.0;
    this.init_weight_magnitude = typeof options.init_weight_magnitude !== 'undefined' ? options.init_weight_magnitude : 1.0;

    this.target_fitness = typeof options.target_fitness !== 'undefined' ? options.target_fitness : 1e20;

    this.debug_mode = typeof options.debug_mode !== 'undefined' ? options.debug_mode : false;

    // module globals should be changed as well
    initMu = 0.0;
    initStdev = this.init_weight_magnitude; // randomised param initialisation.
    mutationRate = this.mutation_rate;
    mutationSize = this.mutation_size;

    // if the below is set to true, then extinction will be turned on for the next evolve()
    this.forceExtinctionMode = false;

    var genome;
    var i, N, K;

    N = this.sub_population_size;
    K = this.num_populations;

    kmedoids.init(K);
    kmedoids.setDistFunction(this.dist);

    this.genes = []; // population
    this.hallOfFame = []; // stores the hall of fame here!
    this.bestOfSubPopulation = []; // stores the best gene for each sub population here.

    this.compressor = new NEATCompressor(); // this guy helps compress the network.

    // generates the initial genomes
    for (i = 0; i < N*K; i++) {

      if (typeof initGenome_ !== 'undefined') {
        genome = new Genome(initGenome_);
      } else {
        genome = new Genome(); // empty one with no connections
      }

      // initially, just create a single connection from input or bias to outputs
      genome.addRandomConnection();
      genome.mutateWeights(1.0, this.mutation_size); // burst mutate init weights

      // stamp meta info into the genome
      genome.fitness = -1e20;
      genome.cluster = R.randi(0, K);
      this.genes.push(genome);

    }

    for (i = 0; i < this.hall_of_fame_size; i++) {
      if (typeof initGenome_ !== 'undefined') {
        genome = new Genome(initGenome_); // don't modify old results in hof
      } else {
        genome = new Genome(); // empty one with no connections

        // initially, just create a single connection from input or bias to outputs
        genome.addRandomConnection();
        genome.mutateWeights(1.0, this.mutation_size); // burst mutate init weights
      }

      // stamp meta info into the genome
      genome.fitness = -1e20;
      genome.cluster = 0; //R.randi(0, K);
      this.hallOfFame.push(genome);
    }


  };

  NEATTrainer.prototype = {
    sortByFitness: function(c) {
      c = c.sort(function (a, b) {
        if (a.fitness > b.fitness) { return -1; }
        if (a.fitness < b.fitness) { return 1; }
        return 0;
      });
    },
    forceExtinction: function() {
      this.forceExtinctionMode = true;
    },
    resetForceExtinction: function() {
      this.forceExtinctionMode = false;
    },
    applyMutations: function(g) {
      // apply mutations (new node, new connection, mutate weights) on a specified genome g
      if (Math.random() < this.new_node_rate) g.addRandomNode();
      if (Math.random() < this.new_connection_rate) g.addRandomConnection();
      g.mutateWeights(this.mutation_rate, this.mutation_size);
    },
    applyFitnessFuncToList: function(f, geneList) {
      var i, n;
      var g;
      for (i=0,n=geneList.length;i<n;i++) {
        g = geneList[i];
        g.fitness = f(g);
      }
    },
    getAllGenes: function() {
      // returns the list of all the genes plus hall(s) of fame
      return this.genes.concat(this.hallOfFame).concat(this.bestOfSubPopulation);
    },
    applyFitnessFunc: function(f, _clusterMode) {
      // applies fitness function f on everyone including hall of famers
      // in the future, have the option to avoid hall of famers
      var i, n;
      var j, m;
      var g;
      var K = this.num_populations;

      var clusterMode = true; // by default, we would cluster stuff (takes time)
      if (typeof _clusterMode !== 'undefined') {
        clusterMode = _clusterMode;
      }

      this.applyFitnessFuncToList(f, this.genes);
      this.applyFitnessFuncToList(f, this.hallOfFame);
      this.applyFitnessFuncToList(f, this.bestOfSubPopulation);

      this.filterFitness();
      this.genes = this.genes.concat(this.hallOfFame);
      this.genes = this.genes.concat(this.bestOfSubPopulation);
      this.sortByFitness(this.genes);

      // cluster before spinning off hall of fame:
      if (clusterMode) {
        this.cluster();
      }

      // rejig hall of fame
      this.hallOfFame = [];
      for (i=0,n=this.hall_of_fame_size;i<n;i++) {
        g = this.genes[i].copy();
        g.fitness = this.genes[i].fitness;
        g.cluster = this.genes[i].cluster;
        this.hallOfFame.push(g);
      }

      // store best of each sub population (may be overlaps with hall of fame)
      this.bestOfSubPopulation = [];
      for (j=0;j<K;j++) {
        for (i=0,n=this.genes.length;i<n;i++) {
          if (this.genes[i].cluster === j) {
            g = this.genes[i].copy();
            g.fitness = this.genes[i].fitness;
            g.cluster = this.genes[i].cluster;
            this.bestOfSubPopulation.push(g);
            break;
          }
        }
      }

    },
    clipWeights: function(maxWeight_) {
      // applies fitness function f on everyone including hall of famers
      // in the future, have the option to avoid hall of famers
      var i, n;
      var g;
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        g.clipWeights(maxWeight_);
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        g.clipWeights(maxWeight_);
      }
    },
    areWeightsNaN: function() {
      // if any weight value is NaN of any gene, then returns true and break.
      var i, n;
      var g;
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        if (g.areWeightsNaN()) return true;
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        if (g.areWeightsNaN()) return true;
      }
      return false;
    },
    filterFitness: function() {
      // achieves 2 things. converts NaN to -1e20
      // makes sure all fitness numbers have negative values
      // makes sure each fitness number is less than minus epsilon
      // the last point is important, since in NEAT, we will randomly choose
      // parents based on their inverse fitness normalised probabilities

      var i, n;
      var epsilon = 1e-10;
      var g;
      function tempProcess(g) {
        var fitness = -1e20;
        if (typeof g.fitness !== 'undefined' && isNaN(g.fitness) === false) {
          fitness = -Math.abs(g.fitness);
          fitness = Math.min(fitness, -epsilon);
        }
        g.fitness = fitness;
      }
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        tempProcess(g);
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        tempProcess(g);
      }
    },
    pickRandomIndex: function(genes, cluster_) {
      // Ehis function returns a random index for a given gene array 'genes'
      // Each element of genes will have a strictly negative .fitness parameter
      // the picking will be probability weighted to the fitness
      // A .normFitness parameter will be tagged onto each element
      // If cluster_ is specified, then each element of genes will be assumed
      // to have a .cluster parameter, and the resulting index will be from
      // that cluster.  if not, then all elements will be elegible
      // Assumes that filterFitness is run to clean the .fitness values up.
      var i, n;
      var byCluster = false;
      var cluster = 0;
      var totalProb = 0;
      var g;

      if (typeof cluster_ !== 'undefined') {
        byCluster = true;
        cluster = cluster_;
      }
      n = genes.length;

      var slack = 0.01; // if this is set to > 0, it ensure that the very best solution won't be picked each time.
      // if it is 0, and the best solution has a solution closer t -0, then best solution will be picked each time.

      // create inverse fitnesses
      for (i=0;i<n;i++) {
        g = genes[i];
        if (byCluster === false || g.cluster === cluster) {
          g.normFitness = 1/(-g.fitness+slack);
          //g.normFitness *= g.normFitness; // square this bitch, so crappy solutions have less chances...
          totalProb += g.normFitness;
        }
      }

      // normalize each fitness
      for (i=0;i<n;i++) {
        g = genes[i];
        if (byCluster === false || g.cluster === cluster) {
          g.normFitness /= totalProb;
        }
      }

      var x = Math.random(); // x will be [0, 1)
      var idx = -1;

      // find the index that corresponds to probability x
      for (i=0;i<n;i++) {
        g = genes[i];
        if (byCluster === false || g.cluster === cluster) {
          x -= g.normFitness;
          if (x <= 0) {
            idx = i;
            break;
          }
        }
      }

      return idx;
    },
    cluster: function(genePool_) {
      // run K-medoids algorithm to cluster this.genes
      var genePool = this.genes;
      var i, j, m, n, K, idx;
      if (typeof genePool_ !== 'undefined') genePool = genePool_;
      kmedoids.partition(this.genes); // split into K populations

      K = this.num_populations;

      var clusterIndices = kmedoids.getCluster();

      // put everything into new gene clusters
      for (i=0;i<K;i++) {
        m = clusterIndices[i].length;
        for (j=0;j<m;j++) {
          idx = clusterIndices[i][j];
          genePool[idx].cluster = i;
        }
      }
    },
    evolve: function(_mutateWeightsOnly) {
      // this is where the magic happens!
      //
      // performs one step evolution of the entire population
      //
      // assumes that applyFitnessFunc() or .fitness vals have been populated
      // .fitness values must be strictly negative.

      // concats both genes and hallOfFame into a combined bigger genepool
      //var prevGenes = this.genes.concat(this.hallOfFame);

      // assumes that clustering is already done!  important.
      // so assumes that the .cluster value for each genome is assigned.
      var prevGenes = this.genes;
      var newGenes = []; // new population array
      var i, n, j, m, K, N, idx;

      var worstFitness = 1e20;
      var worstCluster = -1;

      var bestFitness = -1e20;
      var bestCluster = -1;

      var mutateWeightsOnly = false;

      if (typeof _mutateWeightsOnly !== 'undefined') {
        mutateWeightsOnly = _mutateWeightsOnly;
      }

      K = this.num_populations;
      N = this.sub_population_size;

      // increase the generaiton:
      incrementGenerationCounter();

      var clusterIndices = kmedoids.getCluster();

      var cluster = new Array(K);

      // put everything into new gene clusters
      for (i=0;i<K;i++) {
        m = clusterIndices[i].length;
        cluster[i] = new Array(m);
        for (j=0;j<m;j++) {
          idx = clusterIndices[i][j];
          cluster[i][j] = prevGenes[idx];
        }
        this.sortByFitness(cluster[i]);

        // determine worst cluster (to destroy that sub population)
        if (cluster[i][0].fitness < worstFitness) {
          worstFitness = cluster[i][0].fitness;
          worstCluster = i;
        }

        // determine best cluster
        if (cluster[i][0].fitness >= bestFitness) {
          bestFitness = cluster[i][0].fitness;
          bestCluster = i;
        }
      }

      var mom, dad, baby, momIdx, dadIdx;

      // whether to kill off crappiest sub population and replace with best sub population
      // if we are just evolving weights only (CNE) then no need for extinction.
      var extinctionEvent = false;
      if (Math.random() < this.extinction_rate && mutateWeightsOnly === false) {
        extinctionEvent = true;
        if (this.debug_mode) console.log('the crappiest sub population will be made extinct now!');
      }
      if (this.forceExtinctionMode && mutateWeightsOnly === false) {
        if (this.debug_mode) console.log('forced extinction of crappiest sub population.');
        extinctionEvent = true;
      }

      for (i=0;i<K;i++) {

        // go thru each cluster, and mate N times with 2 random parents each time
        // if it is the worst cluster, then use everything.
        for (j=0;j<N;j++) {
          if (extinctionEvent && i === worstCluster) {
            momIdx = this.pickRandomIndex(prevGenes,bestCluster);
            dadIdx = this.pickRandomIndex(prevGenes,bestCluster);
          } else {
            momIdx = this.pickRandomIndex(prevGenes,i);
            dadIdx = this.pickRandomIndex(prevGenes,i);
          }
          mom = prevGenes[momIdx];
          dad = prevGenes[dadIdx];

          try {

          if (mutateWeightsOnly) {
            baby = mom.crossover(dad);
            //baby = mom.copy();
            baby.mutateWeights(this.mutation_rate, this.mutation_size);
          } else {
            baby = mom.crossover(dad);
            this.applyMutations(baby);
          }

          } catch (err) {
            if (this.debug_mode) {
              console.log("Error with mating: "+err);
              console.log("momIdx = "+momIdx);
              console.log("dadIdx = "+dadIdx);
              console.log("mom:");
              console.log(mom);
              console.log("dad:");
              console.log(dad);
            }
            baby = this.getBestGenome(i).copy();
            this.applyMutations(baby);
          }
          finally {
            baby.cluster = i;
            newGenes.push(baby);
          }
        }
      }

      this.genes = newGenes;

      this.compressor.buildMap(this.getAllGenes());
      this.compressor.compressNEAT();
      this.compressor.compressGenes(this.genes);
      this.compressor.compressGenes(this.hallOfFame);
      this.compressor.compressGenes(this.bestOfSubPopulation);

    },
    printFitness: function() {
      // debug function to print out fitness for all genes and hall of famers
      var i, n;
      var g;
      for (i=0,n=this.genes.length;i<n;i++) {
        g = this.genes[i];
        console.log('genome '+i+' fitness = '+g.fitness);
      }
      for (i=0,n=this.hallOfFame.length;i<n;i++) {
        g = this.hallOfFame[i];
        console.log('hallOfFamer '+i+' fitness = '+g.fitness);
      }
      for (i=0,n=this.bestOfSubPopulation.length;i<n;i++) {
        g = this.bestOfSubPopulation[i];
        console.log('bestOfSubPopulation '+i+' fitness = '+g.fitness);
      }
    },
    getBestGenome: function(cluster_) {
      // returns the b
      var bestN = 0;
      var cluster = 0;
      var i, n;
      var g;

      var allGenes = this.genes;
      this.sortByFitness(allGenes);
      if (typeof cluster_ === 'undefined') {
        return allGenes[bestN];
      }
      cluster = cluster_;
      for (i=0,n=allGenes.length;i<n;i++) {
        g = allGenes[i];
        if (g.cluster === cluster) {
          bestN = i;
          break;
        }
      }
      return allGenes[bestN];
    },
    dist: function(g1, g2) { // calculates 'distance' between 2 genomes
      g1.createUnrolledConnections();
      g2.createUnrolledConnections();

      var coef = { // coefficients for determining distance
        excess : 10.0,
        disjoint : 10.0,
        weight : 0.1,
      };

      var i, n, c1, c2, exist1, exist2, w1, w2, lastIndex1, lastIndex2, minIndex, active1, active2;
      //var active1, active2;
      var nBothActive = 0;
      var nDisjoint = 0;
      var nExcess = 0;
      var weightDiff = 0;
      var unrolledConnections1 = [];
      var unrolledConnections2 = [];
      n=connections.length; // global connection length

      var diffVector = R.zeros(n);

      for (i=0;i<n;i++) {
        c1 = g1.unrolledConnections[i];
        c2 = g2.unrolledConnections[i];
        exist1 = c1[IDX_CONNECTION];
        exist2 = c2[IDX_CONNECTION];
        active1 = exist1*c1[IDX_ACTIVE];
        active2 = exist2*c2[IDX_ACTIVE];
        if (exist1 === 1) lastIndex1 = i;
        if (exist2 === 1) lastIndex2 = i;
        diffVector[i] = (exist1 === exist2)? 0 : 1; // record if one is active and the other is not
        if (active1 === 1 && active2 === 1) { // both active (changed to exist)
          w1 = c1[IDX_WEIGHT];
          w2 = c2[IDX_WEIGHT];
          R.assert(!isNaN(w1), 'weight1 inside dist function is NaN.');
          R.assert(!isNaN(w2), 'weight2 inside dist function is NaN.');
          nBothActive += 1;
          weightDiff += Math.abs(w1 - w2);
        }
      }
      minIndex = Math.min(lastIndex1, lastIndex2);
      if (nBothActive > 0) weightDiff /= nBothActive; // calculate average weight diff

      for (i=0;i<=minIndex;i++) {
        // count disjoints
        nDisjoint += diffVector[i];
      }
      for (i=minIndex+1;i<n;i++) {
        // count excess
        nExcess += diffVector[i];
      }

      var numNodes = Math.max(g1.getNodesInUse().length,g2.getNodesInUse().length);
      var distDisjoint = coef.disjoint*nDisjoint / numNodes;
      var distExcess = coef.excess*nExcess / numNodes;
      var distWeight = coef.weight * weightDiff;
      var distance = distDisjoint+distExcess+distWeight;

      if (isNaN(distance) || Math.abs(distance) > 100) {
        console.log('large distance report:');
        console.log('distance = '+distance);
        console.log('disjoint = '+distDisjoint);
        console.log('excess = '+distExcess);
        console.log('weight = '+distWeight);
        console.log('numNodes = '+numNodes);
        console.log('nBothActive = '+nBothActive);
      }

      /*
      console.log('distance calculation');
      console.log('nDisjoint = '+nDisjoint);
      console.log('nExcess = '+nExcess);
      console.log('avgWeightDiff = '+weightDiff);
      console.log('distance = '+distance);
      */

      return distance;
    },
  };

  global.init = init;
  global.Genome = Genome;
  global.getNodes = getNodes;
  global.getConnections = getConnections;
  global.randomizeRenderMode = randomizeRenderMode;
  global.setRenderMode = setRenderMode;
  global.getRenderMode = getRenderMode;
  global.NEATTrainer = NEATTrainer;
  global.NEATCompressor = NEATCompressor;
  global.getNumInput = function() { return nInput; };
  global.getNumOutput = function() { return nOutput; };
  global.incrementGenerationCounter = incrementGenerationCounter;
  global.getNumGeneration = function() { return generationNum; };


})(N);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(N);



},{"./api.js":1,"./kmedoids.js":2,"./recurrent.js":4}],4:[function(require,module,exports){
// MIT License

// heavily modified recurrent.js library for use in genetic art
// with DCT compression, CoSyNe neuroevolution
// sin, cos, gaussian, abs activations

// based off https://github.com/karpathy/recurrentjs, excellent library by karpathy

var R = {};

(function(global) {
  "use strict";

  // Utility fun
  function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // Random numbers utils
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) {
      return_v = false;
      return v_val;
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r === 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  };
  var randf = function(a, b) { return Math.random()*(b-a)+a; };
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); };
  var randn = function(mu, std){ return mu+gaussRandom()*std; };

  // helper function returns array of zeros of length n
  // and uses typed arrays if available
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i] = 0; }
      return arr;
    } else {
      if (n > 150000) console.log('creating a float array of length = '+n);
      //return new Float64Array(n);
      return new Float32Array(n);
    }
  };

  var copy = function(floatArray) {
    // returns a copy of floatArray
    var n = floatArray.length;
    var result = zeros(n);
    for (var i=0;i<n;i++) {
      result[i]=floatArray[i];
    }
    return result;
  };

  var shuffle = function(origArray) {
    // returns a newArray which is a shuffled version of origArray
    var i, randomIndex;
    var temp;
    var N = origArray.length;
    var result = zeros(N);
    for (i=0;i<N;i++) {
      result[i] = origArray[i];
    }
    for (i=0;i<N;i++) {
      // swaps i with randomIndex
      randomIndex = randi(0, N);
      temp = result[randomIndex];
      result[randomIndex] = result[i];
      result[i] = temp;
    }
    return result;
  };

  // Mat holds a matrix
  var Mat = function(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  };
  Mat.prototype = {
    get: function(row, col) {
      // slow but careful accessor function
      // we want row-major order
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      return this.w[ix];
    },
    set: function(row, col, v) {
      // slow but careful accessor function
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      this.w[ix] = v;
    },
    setAll: function(v) {
      // sets all value of Mat (.w) to v
      var i, n;
      for (i=0,n=this.n*this.d;i<n;i++) {
        this.w[i] = v;
      }
    },
    setFromArray: function(a) {
      var i, j;
      assert(this.n === a.length && this.d === a[0].length);
      for (i=0;i<this.n;i++) {
        for (j=0;j<this.d;j++) {
          this.set(i, j, a[i][j]);
        }
      }
    },
    copy: function() {
      // return a copy of Mat
      var result = new Mat(this.n, this.d);
      var i, len;
      len = this.n*this.d;
      for (i = 0; i < len; i++) {
        result.w[i] = this.w[i];
        result.dw[i] = this.dw[i];
      }
      return result;
    },
    toString: function(precision_) {
      var result_w = '[';
      var i, j;
      var n, d;
      var ix;
      var precision = 10e-4 || precision_;
      precision = 1/precision;
      n = this.n;
      d = this.d;
      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          ix = i*d+j;
          assert(ix >= 0 && ix < this.w.length);
          result_w+=''+Math.round(precision*this.w[ix])/precision+',\t';
        }
        result_w+='\n';
      }
      result_w+=']';
      return result_w;
    },
    print: function() {
      console.log(this.toString());
    },
    dct2: function() {
      // inefficient implementation of discrete cosine transform (2d)
      // ref: http://www.mathworks.com/help/images/ref/dct2.html
      var n = this.n;
      var d = this.d;
      var B = new Mat(n, d); // resulting matrix
      var i, j, k, l;
      var temp;
      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          temp=0;
          for (k=0;k<n;k++) {
            for (l=0;l<d;l++) {
              temp=temp+this.w[k*d+l]*Math.cos(i*Math.PI*((2*k)+1)/(2*n))*Math.cos(j*Math.PI*((2*l)+1)/(2*d));
            }
          }
          if ((i===0)&&(j!==0)) temp*=1/Math.SQRT2;
          if ((j===0)&&(i!==0)) temp*=1/Math.SQRT2;
          if ((j===0)&&(i===0)) temp*=0.5;
          B.w[i*d+j]=temp*2/Math.sqrt(n*d);
        }
      }
      return B;
    },
    idct2: function() {
      // inefficient implementation of inverse discrete cosine transform (2d)
      // ref: http://www.mathworks.com/help/images/ref/idct2.html
      var n = this.n;
      var d = this.d;
      var A = new Mat(n, d); // resulting matrix
      var i, j, k, l;
      var temp;
      for (i=0;i<n;i++) {
        for (j=0;j<d;j++) {
          A.w[i*d+j]=0;
          for (k=0;k<n;k++) {
            for (l=0;l<d;l++) {
              temp=this.w[k*d+l]*Math.cos((((2*i)+1)*k*Math.PI)/(2*n))*Math.cos((((2*j)+1)*l*Math.PI)/(2*d));

              if ((k===0)&&(l===0)) temp*=0.5;
              if ((k!==0)&&(l===0)) temp*=1/Math.SQRT2;
              if ((k===0)&&(l!==0)) temp*=1/Math.SQRT2;

              A.w[i*d+j]+=temp*2/Math.sqrt(n*d);

            }
          }
        }
      }
      return A;
    },
    toJSON: function() {
      var json = {};
      json.n = this.n;
      json.d = this.d;
      json.w = this.w;
      return json;
    },
    fromJSON: function(json) {
      this.n = json.n;
      this.d = json.d;
      this.w = zeros(this.n * this.d);
      this.dw = zeros(this.n * this.d);
      for(var i=0,n=this.n * this.d;i<n;i++) {
        this.w[i] = json.w[i]; // copy over weights
      }
    }
  };

  // return Mat but filled with random numbers from gaussian
  var RandMat = function(n,d,mu,std) {
    var m = new Mat(n, d);
    fillRandn(m,mu || 0,std || 0.08); // kind of :P
    return m;
  };

  // Mat utils
  // fill matrix with random gaussian numbers
  var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } };
  var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } };

  // Transformer definitions
  var Graph = function(needs_backprop) {
    if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = [];
  };
  Graph.prototype = {
    backward: function() {
      // for(var i=this.backprop.length-1;i>=0;i--) {
      //   this.backprop[i](); // tick!
      // }
    },
    rowPluck: function(m, ix) {
      // pluck a row of m with index ix and return it as col vector
      assert(ix >= 0 && ix < m.n);
      var d = m.d;
      var out = new Mat(d, 1);
      for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    sin: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.sin(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sin(x) is cos(x)
            var mwi = out.w[i];
            m.dw[i] += Math.cos(m.w[i]) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    cos: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.cos(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sin(x) is cos(x)
            var mwi = out.w[i];
            m.dw[i] += -Math.sin(m.w[i]) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    gaussian: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      //var c = (1.0/Math.sqrt(2*Math.PI)); // constant of 1 / sqrt(2*pi)
      var c = 1.0; // make amplitude bigger than normal gaussian
      for(var i=0;i<n;i++) {
        out.w[i] = c*Math.exp(-m.w[i]*m.w[i]/2);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sin(x) is cos(x)
            var mwi = out.w[i];
            m.dw[i] += -m.w[i] * mwi * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    tanh: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.tanh(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    sigmoid: function(m) {
      // sigmoid nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = sig(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = sigmoid(x) is z(1 - z)
            var mwi = out.w[i];
            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    relu: function(m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.max(0, m.w[i]); // relu
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    abs: function(m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) {
        out.w[i] = Math.abs(m.w[i]); // relu
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : -out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    mul: function(m1, m2) {
      // multiply matrices m1 * m2
      assert(m1.d === m2.n, 'matmul dimensions misaligned');

      var n = m1.n;
      var d = m2.d;
      var out = new Mat(n,d);
      for(var i=0;i<m1.n;i++) { // loop over rows of m1
        for(var j=0;j<m2.d;j++) { // loop over cols of m2
          var dot = 0.0;
          for(var k=0;k<m1.d;k++) { // dot product loop
            dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
          }
          out.w[d*i+j] = dot;
        }
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<m1.n;i++) { // loop over rows of m1
            for(var j=0;j<m2.d;j++) { // loop over cols of m2
              for(var k=0;k<m1.d;k++) { // dot product loop
                var b = out.dw[d*i+j];
                m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
                m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
              }
            }
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    add: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] + m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += out.dw[i];
            m2.dw[i] += out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
    eltmul: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] * m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += m2.w[i] * out.dw[i];
            m2.dw[i] += m1.w[i] * out.dw[i];
          }
        };
        this.backprop.push(backward);
      }
      return out;
    },
  };

  var softmax = function(m) {
      var out = new Mat(m.n, m.d); // probability volume
      var maxval = -999999;
      var i, n;
      for(i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

      var s = 0.0;
      for(i=0,n=m.w.length;i<n;i++) {
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
      }
      for(i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

      // no backward pass here needed
      // since we will use the computed probabilities outside
      // to set gradients directly on m
      return out;
  };

  var Solver = function() {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  };
  Solver.prototype = {
    step: function(model, step_size, regc, clipval) {
      // // perform parameter update
      // var solver_stats = {};
      // var num_clipped = 0;
      // var num_tot = 0;
      // for(var k in model) {
      //   if(model.hasOwnProperty(k)) {
      //     var m = model[k]; // mat ref
      //     if(!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
      //     var s = this.step_cache[k];
      //     for(var i=0,n=m.w.length;i<n;i++) {

      //       // rmsprop adaptive learning rate
      //       var mdwi = m.dw[i];
      //       if (isNaN(mdwi)) {
      //         /*
      //         console.log('backprop has numerical issues.');
      //         console.log('dWeight '+i+' is NaN');
      //         console.log('setting dw to zero');
      //         */
      //         m.dw[i] = 0.0;
      //         mdwi = 0.0;
      //       }
      //       s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

      //       // gradient clip
      //       if(mdwi > clipval) {
      //         mdwi = clipval;
      //         num_clipped++;
      //       }
      //       if(mdwi < -clipval) {
      //         mdwi = -clipval;
      //         num_clipped++;
      //       }
      //       num_tot++;

      //       if ((s.w[i] + this.smooth_eps) <= 0) {
      //         console.log('rmsprop has numerical issues');
      //         console.log('step_cache '+i+' = '+s.w[i]);
      //         console.log('smooth_eps = '+this.smooth_eps);
      //       }

      //       // update (and regularize)
      //       m.w[i] += - step_size * mdwi / Math.sqrt(Math.max(s.w[i],this.smooth_eps)) - regc * m.w[i];
      //       m.dw[i] = 0; // reset gradients for next iteration

      //       // clip the actual weights as well
      //       if(m.w[i] > clipval*10) {
      //         //console.log('rmsprop clipped the weight with orig value '+m.w[i]);
      //         m.w[i] = clipval*10;
      //       } else if(m.w[i] < -clipval*10) {
      //         //console.log('rmsprop clipped the weight with orig value '+m.w[i]);
      //         m.w[i] = -clipval*10;
      //       }

      //       assert(!isNaN(m.w[i]), 'weight '+i+' is NaN');

      //     }
      //   }
      // }
      // solver_stats.ratio_clipped = num_clipped*1.0/num_tot;
      // return solver_stats;
      return { ratio_clipped: 0 }; 
    }
  };

  var initLSTM = function(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    var hidden_size;
    var prev_size;
    for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
      prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      hidden_size = hidden_sizes[d];

      // gates parameters
      model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bi'+d] = new Mat(hidden_size, 1);
      model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bf'+d] = new Mat(hidden_size, 1);
      model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bo'+d] = new Mat(hidden_size, 1);
      // cell write params
      model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);
      model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bc'+d] = new Mat(hidden_size, 1);
    }
    // decoder params
    model.Whd = new RandMat(output_size, hidden_size, 0, 0.08);
    model.bd = new Mat(output_size, 1);
    return model;
  };

  var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    var hidden_prevs;
    var cell_prevs;
    var d;

    if(typeof prev.h === 'undefined') {
      hidden_prevs = [];
      cell_prevs = [];
      for(d=0;d<hidden_sizes.length;d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
        cell_prevs.push(new R.Mat(hidden_sizes[d],1));
      }
    } else {
      hidden_prevs = prev.h;
      cell_prevs = prev.c;
    }

    var hidden = [];
    var cell = [];
    for(d=0;d<hidden_sizes.length;d++) {

      var input_vector = d === 0 ? x : hidden[d-1];
      var hidden_prev = hidden_prevs[d];
      var cell_prev = cell_prevs[d];

      // input gate
      var h0 = G.mul(model['Wix'+d], input_vector);
      var h1 = G.mul(model['Wih'+d], hidden_prev);
      var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

      // forget gate
      var h2 = G.mul(model['Wfx'+d], input_vector);
      var h3 = G.mul(model['Wfh'+d], hidden_prev);
      var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

      // output gate
      var h4 = G.mul(model['Wox'+d], input_vector);
      var h5 = G.mul(model['Woh'+d], hidden_prev);
      var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

      // write operation on cells
      var h6 = G.mul(model['Wcx'+d], input_vector);
      var h7 = G.mul(model['Wch'+d], hidden_prev);
      var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

      // compute new cell activation
      var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
      var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
      var cell_d = G.add(retain_cell, write_cell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

      hidden.push(hidden_d);
      cell.push(cell_d);
    }

    // one decoder to outputs at end
    var output = G.add(G.mul(model.Whd, hidden[hidden.length - 1]),model.bd);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'c':cell, 'o' : output};
  };

  var initRNN = function(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    var hidden_size, prev_size;
    for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
      prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      hidden_size = hidden_sizes[d];
      model['Wxh'+d] = new R.RandMat(hidden_size, prev_size , 0, 0.08);
      model['Whh'+d] = new R.RandMat(hidden_size, hidden_size, 0, 0.08);
      model['bhh'+d] = new R.Mat(hidden_size, 1);
    }
    // decoder params
    model.Whd = new RandMat(output_size, hidden_size, 0, 0.08);
    model.bd = new Mat(output_size, 1);
    return model;
  };

  var forwardRNN = function(G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of RNN
    // G is graph to append ops to
    // model contains RNN parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden activations from last step

    var hidden_prevs;
    var d;
    if(typeof prev.h === 'undefined') {
      hidden_prevs = [];
      for(d=0;d<hidden_sizes.length;d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d],1));
      }
    } else {
      hidden_prevs = prev.h;
    }

    var hidden = [];
    for(d=0;d<hidden_sizes.length;d++) {

      var input_vector = d === 0 ? x : hidden[d-1];
      var hidden_prev = hidden_prevs[d];

      var h0 = G.mul(model['Wxh'+d], input_vector);
      var h1 = G.mul(model['Whh'+d], hidden_prev);
      var hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh'+d]));

      hidden.push(hidden_d);
    }

    // one decoder to outputs at end
    var output = G.add(G.mul(model.Whd, hidden[hidden.length - 1]),model.bd);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'o' : output};
  };

  var sig = function(x) {
    // helper function for computing sigmoid
    return 1.0/(1+Math.exp(-x));
  };

  var maxi = function(w) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for(var i=1,n=w.length;i<n;i++) {
      var v = w[i];
      if(v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  };

  var samplei = function(w) {
    // sample argmax from w, assuming w are
    // probabilities that sum to one
    var r = randf(0,1);
    var x = 0.0;
    var i = 0;
    while(true) {
      x += w[i];
      if(x > r) { return i; }
      i++;
    }
    return w.length - 1; // pretty sure we should never get here?
  };

  var getModelSize = function(model) {
    // returns the size (ie, number of floats) used in a model
    var len = 0;
    var k;
    var m;
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len += m.w.length;
      }
    }
    return len;
  };

  // other utils
  var flattenModel = function(model, gradient_) {
    // returns an float array containing a dump of model's params in order
    // if gradient_ is true, the the flatten model returns the dw's, rather whan w's.
    var len = 0; // determine length of dump
    var i = 0;
    var j;
    var k;
    var m;
    var gradientMode = false || gradient_;
    len = getModelSize(model);
    var result = R.zeros(len);
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (j = 0; j < len; j++) {
          if (gradientMode) {
            result[i] = m.dw[j];
          } else {
            result[i] = m.w[j];
          }
          i++;
        }
      }
    }
    return result;
  };
  var pushToModel = function(model, dump, gradient_) {
    // pushes a float array containing a dump of model's params into a model
    // if gradient_ is true, dump will be pushed to dw's, rather whan w's.
    var len = 0; // determine length of dump
    var i = 0;
    var j;
    var k;
    var m;
    var gradientMode = false || gradient_;
    len = getModelSize(model);
    assert(dump.length === len); // make sure the array dump has same len as model
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (j = 0; j < len; j++) {
          if (gradientMode) {
            m.dw[j] = dump[i];
          } else {
            m.w[j] = dump[i];
          }
          i++;
        }
      }
    }
  };
  var copyModel = function(model) {
    // returns an exact copy of a model
    var k;
    var m;
    var result = [];
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        result[k] = m.copy();
      }
    }
    return result;
  };
  var zeroOutModel = function(model, gradientAsWell_) {
    // zeros out every element (including dw, if gradient_ is true) of model
    var len = 0; // determine length of dump
    var j;
    var k;
    var m;
    var gradientMode = false || gradientAsWell_;

    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (j = 0; j < len; j++) {
          if (gradientMode) {
            m.dw[j] = 0;
          }
          m.w[j] = 0;
        }
      }
    }
  };
  var compressModel = function(model, nCoef) {
    // returns a compressed model using 2d-dct
    // each model param will be compressed down to:
    // min(nRow, nCoef), min(nCol, nCoef)
    var k;
    var m;
    var nRow, nCol;
    var result = [];
    var z; // dct transformed matrix
    var i, j;
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        z = m.dct2();
        nRow = Math.min(z.n, nCoef);
        nCol = Math.min(z.d, nCoef);
        result[k] = new Mat(nRow, nCol);
        for (i=0;i<nRow;i++) {
          for (j=0;j<nCol;j++) {
            result[k].set(i, j, z.get(i, j));
          }
        }
      }
    }
    return result;
  };
  var decompressModel = function(small, model) {
    // decompresses small (a compressed model) into model using idct
    var k;
    var m, s;
    var nRow, nCol;
    var z; // idct transformed matrix
    var i, j;
    for(k in small) {
      if(small.hasOwnProperty(k)) {
        s = small[k];
        m = model[k];
        nRow = m.n;
        nCol = m.d;
        z = new Mat(nRow, nCol);
        for (i=0;i<s.n;i++) {
          for (j=0;j<s.d;j++) {
            z.set(i, j, s.get(i, j));
          }
        }
        model[k] = z.idct2();
      }
    }
  };
  var numGradient = function(f, model, avgDiff_, epsilon_) {
    // calculates numerical gradient.  fitness f is forward pass function passed in is a function of model only.
    // f will be run many times when the params of each indiv weight changes
    // numerical gradient is computed off average of uptick and downtick gradient, so O(e^2) noise.
    // returns a mat object, where .w holds percentage differences, and .dw holds numerical gradient
    // if avgDiff_ mode is set to true, returns the average percentage diff rather than the actual gradients
    var epsilon = 1e-10 || epsilon_;
    var avgDiff = false || avgDiff_;
    var base = f(model);
    var upBase, downBase;
    var uptick = copyModel(model); // uptick.w holds the ticked weight value
    var downtick = copyModel(model); // opposite of uptick.w
    var numGrad = copyModel(model); // numGrad.dw holds the numerical gradient.

    var avgPercentDiff = 0.0;
    var avgPercentDiffCounter = 0;

    var i, len;
    var m;
    var k;
    var result = [];
    for(k in model) {
      if(model.hasOwnProperty(k)) {
        m = model[k];
        len = m.w.length;
        for (i = 0; i < len; i++) {
          // change the weights by a small amount to find gradient
          uptick[k].w[i] += epsilon;
          downtick[k].w[i] -= epsilon;
          upBase = f(uptick);
          downBase = f(downtick);
          // store numerical gradient
          numGrad[k].dw[i] = (upBase - downBase) / (2 * epsilon);
          numGrad[k].w[i] = ((numGrad[k].dw[i] + epsilon) / (model[k].dw[i] + epsilon) - 1);
          avgPercentDiff += numGrad[k].w[i] * numGrad[k].w[i];
          avgPercentDiffCounter += 1;
          // store precentage diff in w.
          // undo the change of weights by a small amount
          uptick[k].w[i] -= epsilon;
          downtick[k].w[i] += epsilon;

          // set model's dw to numerical gradient (useful for debugging)
          //model[k].dw[i] = numGrad[k].dw[i];
        }
      }
    }

    if (avgDiff) {
      return Math.sqrt(avgPercentDiff / avgPercentDiffCounter);
    }
    return numGrad;
  };

  // neuroevolution tools

  // chromosome implementation using an array of floats
  var Gene = function(initFloatArray) {
    var i;
    var len = initFloatArray.length;
    // the input array will be copied
    this.fitness = -1e20; // default fitness value is very negative
    this.nTrial = 0; // number of trials subjected to so far.
    this.gene = zeros(len);
    for (i=0;i<len;i++) {
      this.gene[i] = initFloatArray[i];
    }
  };

  Gene.prototype = {
    burstMutate: function(burst_magnitude_) { // adds a normal random variable of stdev width, zero mean to each gene.
      var burst_magnitude = burst_magnitude_ || 0.1;
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        this.gene[i] += randn(0.0, burst_magnitude);
      }
    },
    randomize: function(burst_magnitude_) { // resets each gene to a random value with zero mean and stdev
      var burst_magnitude = burst_magnitude_ || 0.1;
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        this.gene[i] = randn(0.0, burst_magnitude);
      }
    },
    mutate: function(mutation_rate_, burst_magnitude_) { // adds random gaussian (0,stdev) to each gene with prob mutation_rate
      var mutation_rate = mutation_rate_ || 0.1;
      var burst_magnitude = burst_magnitude_ || 0.1;
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        if (randf(0,1) < mutation_rate) {
          this.gene[i] += randn(0.0, burst_magnitude);
        }
      }
    },
    crossover: function(partner, kid1, kid2, onePoint) {
      // performs one-point crossover with partner to produce 2 kids
      // edit -> changed to uniform crossover method.
      // assumes all chromosomes are initialised with same array size. pls make sure of this before calling
      assert(this.gene.length === partner.gene.length);
      assert(partner.gene.length === kid1.gene.length);
      assert(kid1.gene.length === kid2.gene.length);
      var onePointMode = false;
      if (typeof onePoint !== 'undefined') onePointMode = onePoint;
      var i, N;
      N = this.gene.length;
      var cross = true;
      var l = randi(0, N); // crossover point (for one point xover)
      for (i = 0; i < N; i++) {
        if (onePointMode) {
          cross = (i < l);
        } else {
          cross = (Math.random() < 0.5);
        }
        if (cross) {
          kid1.gene[i] = this.gene[i];
          kid2.gene[i] = partner.gene[i];
        } else {
          kid1.gene[i] = partner.gene[i];
          kid2.gene[i] = this.gene[i];
        }
      }
    },
    copyFrom: function(g) { // copies g's gene into itself
      var i, N;
      this.copyFromArray(g.gene);
    },
    copyFromArray: function(sourceFloatArray) {
      // copy an array own's gene (must be same size)
      assert(this.gene.length === sourceFloatArray.length);
      var i, N;
      N = this.gene.length;
      for (i = 0; i < N; i++) {
        this.gene[i] = sourceFloatArray[i];
      }
    },
    copy: function(precision_) { // returns a rounded exact copy of itself (into new memory, doesn't return reference)
      var precision = 10e-4 || precision_; precision = 1/precision;
      var newFloatArray = zeros(this.gene.length);
      var i;
      for (i = 0; i < this.gene.length; i++) {
        newFloatArray[i] = Math.round(precision*this.gene[i])/precision;
      }
      var g = new Gene(newFloatArray);
      g.fitness = this.fitness;
      g.nTrial = this.nTrial;
      return g;
    },
    pushToModel: function(model) { // pushes this chromosome to a specified network
      pushToModel(model, this.gene);
    }
  };

  // randomize neural network model with random weights and biases
  var randomizeModel = function(model, magnitude_) {
    var modelSize = getModelSize(model);
    var magnitude = 1.0 || magnitude_;
    var r = new RandMat(1, modelSize, 0, magnitude);
    var g = new Gene(r.w);
    g.pushToModel(model);
  };

  var GATrainer = function(model, options_, init_gene_array) {
    // implementation of CoSyNe neuroevolution framework
    //
    // options:
    // population_size : positive integer
    // hall_of_fame_size : positive integer, stores best guys in all of history and keeps them.
    // mutation_rate : [0, 1], when mutation happens, chance of each gene getting mutated
    // elite_percentage : [0, 0.3], only this group mates and produces offsprings
    // mutation_size : positive floating point.  stdev of gausian noise added for mutations
    // target_fitness : after fitness achieved is greater than this float value, learning stops
    // init_weight_magnitude : stdev of initial random weight (default = 1.0)
    // burst_generations : positive integer.  if best fitness doesn't improve after this number of generations
    //                    then mutate everything!
    // best_trial : default 1.  save best of best_trial's results for each chromosome.
    // num_match : for use in arms race mode.  how many random matches we set for each chromosome when it is its turn.
    //
    // init_gene_array:  init float array to initialize the chromosomes.  can be result obtained from pretrained sessions.
    // debug_mode: false by default.  if set to true, console.log debug output occurs.

    this.model = copyModel(model); // makes a local working copy of the model. copies architecture and weights

    var options = options_ || {};
    this.hall_of_fame_size = typeof options.hall_of_fame_size !== 'undefined' ? options.hall_of_fame_size : 5;
    this.population_size = typeof options.population_size !== 'undefined' ? options.population_size : 30;
    this.population_size += this.hall_of_fame_size; // make room for hall of fame beyond specified population size.
    this.population_size = Math.floor(this.population_size/2)*2; // make sure even number
    this.length = this.population_size; // use the variable length to suit array pattern.
    this.mutation_rate = typeof options.mutation_rate !== 'undefined' ? options.mutation_rate : 0.01;
    this.init_weight_magnitude = typeof options.init_weight_magnitude !== 'undefined' ? options.init_weight_magnitude : 1.0;
    this.elite_percentage = typeof options.elite_percentage !== 'undefined' ? options.elite_percentage : 0.2;
    this.mutation_size = typeof options.mutation_size !== 'undefined' ? options.mutation_size : 0.5;
    this.debug_mode = typeof options.debug_mode !== 'undefined' ? options.debug_mode : false;
    this.gene_size = getModelSize(this.model); // number of floats in each gene

    var initGene;
    var i;
    var gene;
    if (init_gene_array) {
      initGene = new Gene(init_gene_array);
    }

    this.genes = []; // population
    this.hallOfFame = []; // stores the hall of fame here!
    for (i = 0; i < this.population_size; i++) {
      gene = new Gene(zeros(this.gene_size));
      if (initGene) { // if initial gene supplied, burst mutate param.
        gene.copyFrom(initGene);
        if (i > 0) { // don't mutate the first guy.
          gene.burstMutate(this.mutation_size);
        }
      } else {
        gene.randomize(this.init_weight_magnitude);
      }
      this.genes.push(gene);
    }
    // generates first few hall of fame genes (but burst mutates some of them)
    for (i = 0; i < this.hall_of_fame_size; i++) {
      gene = new Gene(zeros(this.gene_size));
      if (init_gene_array) { // if initial gene supplied, burst mutate param.
        gene.copyFrom(initGene);
      } else {
        gene.randomize(this.init_weight_magnitude);
        if (i > 0) { // don't mutate the first guy.
          gene.burstMutate(this.mutation_size);
        }
      }
      this.hallOfFame.push(gene);
    }

    pushToModel(this.model, this.genes[0].gene); // push first chromosome to neural network. (replaced *1 above)

  };

  GATrainer.prototype = {
    sortByFitness: function(c) {
      c = c.sort(function (a, b) {
        if (a.fitness > b.fitness) { return -1; }
        if (a.fitness < b.fitness) { return 1; }
        return 0;
      });
    },
    pushGeneToModel: function(model, i) {
      // pushes the i th gene of the sorted population into a model
      // this ignores hall of fame
      var g = this.genes[i];
      g.pushToModel(model);
    },
    pushBestGeneToModel: function(model) {
      this.pushGeneToModel(model, 0);
    },
    pushHistToModel: function(model, i) {
      // pushes the i th gene of the sorted population into a model from the hall-of-fame
      // this requires hall of fame model to be used
      var Nh = this.hall_of_fame_size;
      assert(Nh > 0); // hall of fame must be used.
      var g = this.hallOfFame[i];
      g.pushToModel(model);
    },
    pushBestHistToModel: function(model) {
      this.pushHistToModel(model, 0);
    },
    flushFitness: function() {
      // resets all the fitness scores to very negative numbers, incl hall-of-fame
      var i, N, Nh;
      var c = this.genes;
      var h = this.hallOfFame;
      N = this.population_size;
      Nh = this.hall_of_fame_size;
      for (i=0;i<N;i++) {
        c[i].fitness = -1e20;
      }
      for (i=0;i<Nh;i++) {
        h[i].fitness = -1e20;
      }

    },
    sortGenes: function() {
      // this functions just sort list of genes by fitness and does not do any
      // cross over or mutations.
      var c = this.genes;
      // sort the chromosomes by fitness
      this.sortByFitness(c);
    },
    evolve: function() {
      // this function does bare minimum simulation of one generation
      // it assumes the code prior to calling evolve would have simulated the system
      // it also assumes that the fitness in each chromosome of this trainer will have been assigned
      // it just does the task of crossovers and mutations afterwards.

      var i, j, N, Nh;
      var c = this.genes;
      var h = this.hallOfFame;

      N = this.population_size;
      Nh = this.hall_of_fame_size;

      // sort the chromosomes by fitness
      this.sortByFitness(c);

      if (this.debug_mode) {
        for (i = 0; i < 5; i++) {
          console.log(i+': '+Math.round(c[i].fitness*100)/100);
        }
        for (i = 5; i >= 1; i--) {
          console.log((N-i)+': '+Math.round(c[N-i].fitness*100)/100);
        }
      }

      // copies best from population to hall of fame:
      for (i = 0; i < Nh; i++) {
        h.push(c[i].copy());
      }

      // sorts hall of fame
      this.sortByFitness(h);
      // cuts off hall of fame to keep only Nh elements
      h = h.slice(0, Nh);

      if (this.debug_mode) {
        console.log('hall of fame:');
        for (i = 0; i < Math.min(Nh, 3); i++) {
          console.log(i+': '+Math.round(h[i].fitness*100)/100);
        }
      }

      // alters population:

      var Nelite = Math.floor(Math.floor(this.elite_percentage*N)/2)*2; // even number
      for (i = Nelite; i < N; i+=2) {
        var p1 = randi(0, Nelite);
        var p2 = randi(0, Nelite);
        c[p1].crossover(c[p2], c[i], c[i+1]);
      }

      // leaves the last Nh slots for hall of fame guys.
      for (i = 0; i < N-Nh; i++) {
        c[i].mutate(this.mutation_rate, this.mutation_size);
      }

      // sneakily puts in the hall of famers back into the population at the end:
      for (i = 0; i < Nh; i++) {
        c[N-Nh+i] = h[i].copy();
      }


      // permutation step in CoSyNe
      // we permute all weights in elite set, and don't prob-weight as in Gomez 2008.

      var Nperm = Nelite; // permute the weights up to Nperm.
      var permData = zeros(Nperm);
      var len = c[0].gene.length; // number of elements in each gene
      for (j=0;j<len;j++) {
        // populate the data to be shuffled
        for (i=0;i<Nperm;i++) {
          permData[i] = c[i].gene[j];
        }
        permData = shuffle(permData); // the magic is supposed to happen here.
        // put back the shuffled data back:
        for (i=0;i<Nperm;i++) {
          c[i].gene[j] = permData[i];
        }
      }

    }
  };

  // various utils
  global.maxi = maxi;
  global.samplei = samplei;
  global.randi = randi;
  global.randf = randf;
  global.randn = randn;
  global.zeros = zeros;
  global.copy = copy;
  global.shuffle = shuffle;
  global.softmax = softmax;
  global.assert = assert;

  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;

  global.forwardLSTM = forwardLSTM;
  global.initLSTM = initLSTM;
  global.forwardRNN = forwardRNN;
  global.initRNN = initRNN;

  // optimization
  global.Solver = Solver;
  global.Graph = Graph;

  // other utils
  global.flattenModel = flattenModel;
  global.getModelSize = getModelSize;
  global.copyModel = copyModel;
  global.zeroOutModel = zeroOutModel;
  global.numGradient = numGradient;
  global.pushToModel = pushToModel;
  global.randomizeModel = randomizeModel;

  // model compression
  global.compressModel = compressModel;
  global.decompressModel = decompressModel;

  // ga
  global.GATrainer = GATrainer;
  global.Gene = Gene;

})(R);
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    //window.jsfeat = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(R);

},{}]},{},[3]);
