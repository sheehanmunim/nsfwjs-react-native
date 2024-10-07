import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Image,
  Button,
  SafeAreaView,
  StyleSheet,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as ImagePicker from "expo-image-picker";
import { decode as atob } from "base-64";
import * as jpeg from "jpeg-js";
import { manipulateAsync } from "expo-image-manipulator";

const modelJson = require("../assets/model/model.json");
const modelWeights = [require("../assets/model/group1-shard1of1.bin")];

const picInputShapeSize = {
  width: 224, // Updated to match model's expected width
  height: 224, // Updated to match model's expected height
};

const NSFW_CLASSES = {
  0: "Drawing",
  1: "Hentai",
  2: "Neutral",
  3: "Porn",
  4: "Sexy",
};

function imageToTensor(rawImageData) {
  const TO_UINT8ARRAY = true;
  const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
  // Drop the alpha channel info for mobilenet
  const buffer = new Uint8Array(width * height * 3);
  let offset = 0; // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset];
    buffer[i + 1] = data[offset + 1];
    buffer[i + 2] = data[offset + 2];

    offset += 4;
  }

  // Normalize the pixel values
  return tf.tidy(() => {
    const tensor = tf.tensor4d(buffer, [1, height, width, 3]);
    return tensor.div(255); // Normalize to 0-1
  });
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: NSFW_CLASSES[topkIndices[i]],
      probability: topkValues[i],
    });
  }
  return topClassesAndProbs;
}

const classify = async (model, img, topk = 5) => {
  if (!model) {
    console.error("Model is not loaded");
    return;
  }
  const logits = model.predict(img);
  const classes = await getTopKClasses(logits, topk);
  logits.dispose();
  return classes;
};

const Index = () => {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [image, setImage] = useState();

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      let model;
      try {
        model = await tf.loadLayersModel(
          bundleResourceIO(modelJson, modelWeights)
        );
      } catch (e) {
        console.log(e);
      }

      setModel(model);
    };
    loadModel();
  }, []);

  const classifyImage = async (uri) => {
    if (!uri) return;
    try {
      const resizedPhoto = await manipulateAsync(
        uri,
        [
          {
            resize: {
              width: picInputShapeSize.width,
              height: picInputShapeSize.height,
            },
          },
        ],
        { format: "jpeg", base64: true }
      );
      const base64 = resizedPhoto.base64;
      const arrayBuffer = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      const imageData = arrayBuffer;
      const imageTensor = imageToTensor(imageData);
      const p = await classify(model, imageTensor);
      setPredictions(p);
      console.log(p);
    } catch (e) {
      console.log(e);
    }
  };

  const onHandlePick = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Image
          source={{ uri: image }}
          style={styles.image}
          onLoad={() => classifyImage(image)}
        />
        <Button title="Pick and Predict" onPress={onHandlePick} />
        {predictions &&
          predictions.map((prediction, index) => (
            <Text key={index} style={styles.predictionText}>
              {prediction.className}:{" "}
              {(prediction.probability * 100).toFixed(2)}%
            </Text>
          ))}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  content: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  image: {
    width: 200,
    height: 200,
    marginBottom: 20,
  },
  predictionText: {
    fontSize: 16,
    marginTop: 10,
  },
});

export default Index;
