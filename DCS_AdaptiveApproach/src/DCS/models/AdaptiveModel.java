package DCS.models;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class AdaptiveModel {

    public static Evaluation buildAndEvaluate(List<SingleClassifierResult> pSingleModelsResult, Instances pTrainingSet, Instances pTestSet, List<Double> fMeasures, String ProjectName, int fold) throws Exception {

        Instances classifierSelectionTrainingSet = new Instances(pTrainingSet);
        Instances classifierSelectionTestSet = new Instances(pTestSet);

        //Add the Best Predictor attribute
        List<String> classifierValues = new ArrayList();
        classifierValues.add("Log");
        classifierValues.add("DTree");
        //classifierValues.add("DTable");
        classifierValues.add("MLP");
        classifierValues.add("RBF");
        classifierValues.add("NB");
        classifierSelectionTrainingSet.insertAttributeAt(new Attribute("bestPredictor", classifierValues), classifierSelectionTrainingSet.numAttributes());
        classifierSelectionTestSet.insertAttributeAt(new Attribute("bestPredictor", classifierValues), classifierSelectionTestSet.numAttributes());
        classifierSelectionTrainingSet.setClassIndex(classifierSelectionTestSet.numAttributes() - 1);
        classifierSelectionTestSet.setClassIndex(classifierSelectionTestSet.numAttributes() - 1);

        //Remove bug as feature for the classifier selection
        classifierSelectionTrainingSet.deleteAttributeAt(classifierSelectionTrainingSet.numAttributes() - 2);
        classifierSelectionTestSet.deleteAttributeAt(classifierSelectionTestSet.numAttributes() - 2);

        //Build the augumentive training set
        for (int i = 0; i < pSingleModelsResult.size(); i++) {
            List<Prediction> classifierPredictions = pSingleModelsResult.get(i).getTrainingSetEvaluation().predictions();
            int bestClassifierOnInstance = 0;
            for (int j = 0; j < classifierPredictions.size(); j++) {
                Prediction value = classifierPredictions.get(j);
                if (value.predicted() == value.actual()) {
                    Instance currentInstance = classifierSelectionTrainingSet.get(j);
                    if (Double.isNaN(currentInstance.value(classifierSelectionTrainingSet.numAttributes() - 1))) {
                        if (!Double.isNaN(fMeasures.get(i))) {
                            currentInstance.setValue(classifierSelectionTrainingSet.numAttributes() - 1, classifierValues.get(i));
                            bestClassifierOnInstance = i;
                        }
                    } else if (fMeasures.get(bestClassifierOnInstance) < fMeasures.get(i)) {
                        if (!Double.isNaN(fMeasures.get(i))) {
                            currentInstance.setValue(classifierSelectionTrainingSet.numAttributes() - 1, classifierValues.get(i));
                            bestClassifierOnInstance = i;
                        }
                    }
                }
            }
        }


        //If none of the classifiers is able to correctly predict an instance, assign the one with the best F-Measure
        for (int j = 0; j < classifierSelectionTrainingSet.size(); j++) {
            Instance currentInstance = classifierSelectionTrainingSet.get(j);
            int bestClassifierOnInstance = 0;
            for (int i = 0; i < pSingleModelsResult.size(); i++) {
                if (Double.isNaN(currentInstance.value(classifierSelectionTrainingSet.numAttributes() - 1))) {
                    if (!Double.isNaN(fMeasures.get(i))) {
                        currentInstance.setValue(classifierSelectionTrainingSet.numAttributes() - 1, classifierValues.get(i));
                        bestClassifierOnInstance = i;
                    }
                } else if (fMeasures.get(bestClassifierOnInstance) < fMeasures.get(i)) {
                    if (!Double.isNaN(fMeasures.get(i))) {
                        currentInstance.setValue(classifierSelectionTrainingSet.numAttributes() - 1, classifierValues.get(i));
                        bestClassifierOnInstance = i;
                    }
                }
            }
        }

        if(fold==0) {
            CSVWriter.generateCSV(new File(ProjectName), classifierSelectionTrainingSet.toArray());
        }

        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(classifierSelectionTrainingSet);

        Evaluation classifierSelectorEvaluation = new Evaluation(classifierSelectionTrainingSet);
        classifierSelectorEvaluation.evaluateModel(randomForest, classifierSelectionTestSet);

        ArrayList<Prediction> classifiersToUse = classifierSelectorEvaluation.predictions();

        AggregateableEvaluation aggregateableEvaluation = null;
        for (int i = 0; i < pTestSet.size(); i++) {
            int classifierToUseIndex = (int) classifiersToUse.get(i).predicted();
            Evaluation evaluation = new Evaluation(pTrainingSet);
            Instances tmpTestSet = new Instances(pTestSet, 1);
            tmpTestSet.add(pTestSet.get(i));
            evaluation.evaluateModel(pSingleModelsResult.get(classifierToUseIndex).getClassifier(), tmpTestSet);
            if (aggregateableEvaluation == null) {
                aggregateableEvaluation = new AggregateableEvaluation(evaluation);
            } else {
                aggregateableEvaluation.aggregate(evaluation);
            }
        }

        return aggregateableEvaluation;
    }

}
