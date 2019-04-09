package DCS;

import DCS.models.AdaptiveModel;
import DCS.models.VotingModel;
import DCS.models.SingleClassifierResult;
import DCS.models.SingleClassifierModel;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Evaluation;

public class RunExperiment {

    public static void main(String[] args) throws IOException, Exception {

        RunExperiment rwe = new RunExperiment();

        //Read Files path from console
        System.out.println("Provide File Path for Data");
        BufferedReader reader1 =
                new BufferedReader(new InputStreamReader(System.in));
        String dataFilePath = reader1.readLine();

        System.out.println("Provide File Path to store result");
        BufferedReader reader2 =
                new BufferedReader(new InputStreamReader(System.in));
        String resultFilePath = reader2.readLine();



        File datasetFolder = new File(dataFilePath);
        File resultFolder = new File(resultFilePath);

        resultFolder.mkdirs();

        ArrayList<String> filePaths = new ArrayList();
        ArrayList<String> fileNames = new ArrayList();

        for (File datasetFile : datasetFolder.listFiles()) {
            filePaths.add(datasetFile.getAbsolutePath());
            fileNames.add(datasetFile.getName());
        }

        List<String> classifierNames = Arrays.asList(new String[]{"Log", "NB", "RBF", "MLP", "DTree"});

        //Create training and test sets
        ValidationTechniques validationTechniques = new ValidationTechniques();

        File csvFile = new File(resultFolder.getAbsolutePath() + File.separator + "performance.csv");
        File compCsvFile = new File(resultFolder.getAbsolutePath() + File.separator + "complementarity.csv");

        try (PrintWriter pw = new PrintWriter(new FileOutputStream(csvFile, false))) {
            pw.write("Project,Model,ACC,PRE,REC,FM,AUC\n");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RunExperiment.class.getName()).log(Level.SEVERE, null, ex);
        }

        try (PrintWriter pw = new PrintWriter(new FileOutputStream(compCsvFile, false))) {
            pw.write("Project,Model1,Model2,common,A-B,B-A\n");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RunExperiment.class.getName()).log(Level.SEVERE, null, ex);
        }

        for (int i = 0; i < filePaths.size(); i++) {

            String filePath = filePaths.get(i);
            String projectName = fileNames.get(i);

            System.out.println("Processing " + filePath);

            List<TrainingTestSet> trainingTestSets = validationTechniques.kFoldsValidation(filePath, 10);

            List<SingleClassifierResult> globalResultList = new ArrayList<>();
            for (int j = 0; j < classifierNames.size() * 10; j++) {
                globalResultList.add(new SingleClassifierResult());
            }

            int classifierCounter = 0;
            List<Double> fMeasures = new ArrayList<>();

            List<AggregateableEvaluation> evaluations = new ArrayList<>();

            for (String classifierName : classifierNames) {
                System.out.println("Classifier " + classifierName);
                AggregateableEvaluation singleEvaluation = null;
                for (int fold = 0; fold < 10; fold++) {
                    TrainingTestSet tts = trainingTestSets.get(fold);
                    SingleClassifierResult singleModelResult = SingleClassifierModel.buildAndEvaluate(classifierName, tts.getTrainingSet(), tts.getTestSet());
                    Evaluation singleFoldEvaluation = singleModelResult.getTestSetEvaluation();
                    globalResultList.set(fold * classifierNames.size() + classifierCounter, singleModelResult);
                    if (singleEvaluation == null) {
                        singleEvaluation = new AggregateableEvaluation(singleFoldEvaluation);
                    } else {
                        singleEvaluation.aggregate(singleFoldEvaluation);
                    }
                }

                evaluations.add(singleEvaluation);

                double fMeasure = rwe.calculateQualityMeasure(csvFile, projectName, classifierName, singleEvaluation);

                fMeasures.add(fMeasure);

                classifierCounter++;
            }

            for (int aCounter = 0; aCounter < classifierNames.size(); aCounter++) {
                AggregateableEvaluation evaluation1 = evaluations.get(aCounter);
                for (int bCounter = 0; bCounter < classifierNames.size(); bCounter++) {
                    if (aCounter != bCounter) {
                        AggregateableEvaluation evaluation2 = evaluations.get(bCounter);
                        int common = 0;
                        int aCorrect = 0;
                        int bCorrect = 0;
                        int size = evaluation1.predictions().size();
                        for (int counter = 0; counter < size; counter++) {
                            double evaluation1Value = evaluation1.predictions().get(counter).predicted();
                            double evaluation2Value = evaluation2.predictions().get(counter).predicted();
                            double actualValue = evaluation1.predictions().get(counter).actual();

                            if (evaluation1Value == evaluation2Value && evaluation1Value == 1.0) {
                                common++;
                            } else if (evaluation1Value == actualValue && evaluation1Value == 1.0) {
                                aCorrect++;
                            } else if (evaluation1Value == 1.0) {
                                bCorrect++;
                            }
                        }
                        try (PrintWriter pw = new PrintWriter(new FileOutputStream(compCsvFile, true))) {
                            pw.write(projectName + "," + classifierNames.get(aCounter) + "," + classifierNames.get(bCounter) + "," + common + "," + aCorrect + "," + bCorrect + "\n");
                        } catch (FileNotFoundException ex) {
                            Logger.getLogger(RunExperiment.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }
                }
            }

            System.out.println("Voting");
            AggregateableEvaluation votingEvaluation = null;
            for (int fold = 0; fold < 10; fold++) {
                TrainingTestSet tts = trainingTestSets.get(fold);
                SingleClassifierResult votingResult = VotingModel.buildAndEvaluate(tts.getTrainingSet(), tts.getTestSet());
                Evaluation votingFoldEvaluation = votingResult.getTestSetEvaluation();
                if (votingEvaluation == null) {
                    votingEvaluation = new AggregateableEvaluation(votingFoldEvaluation);
                } else {
                    votingEvaluation.aggregate(votingFoldEvaluation);
                }
            }
            rwe.calculateQualityMeasure(csvFile, projectName, "Voting", votingEvaluation);

            System.out.println("Adaptive");
            AggregateableEvaluation globalAdaptiveEvaluation = null;
            for (int fold = 0; fold < 10; fold++) {
                TrainingTestSet tts = trainingTestSets.get(fold);
                String logLocation = resultFilePath+"\\"+projectName;
                Evaluation globalAdaptiveEvaluationFold = AdaptiveModel.buildAndEvaluate(globalResultList.subList(fold * classifierNames.size(), (fold + 1) * classifierNames.size()), tts.getTrainingSet(), tts.getTestSet(), fMeasures, logLocation, fold);
                if (globalAdaptiveEvaluation == null) {
                    globalAdaptiveEvaluation = new AggregateableEvaluation(globalAdaptiveEvaluationFold);
                } else {
                    globalAdaptiveEvaluation.aggregate(globalAdaptiveEvaluationFold);
                }
            }
            rwe.calculateQualityMeasure(csvFile, projectName, "Adaptive ", globalAdaptiveEvaluation);
        }
    }

    private double calculateQualityMeasure(File csvFile, String projectName, String modelName, Evaluation evaluation) {
        double truePositive = evaluation.numTruePositives(0);
        double trueNegative = evaluation.numTrueNegatives(0);
        double falsePositive = evaluation.numFalsePositives(0);
        double falseNegative = evaluation.numFalseNegatives(0);

        double accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative);
        double precision = truePositive / (truePositive + falsePositive);
        double recall = truePositive / (truePositive + falseNegative);
        double fmeasure = 2 * precision * recall / (precision + recall);
        double aucroc = evaluation.areaUnderROC(0);

        if (aucroc < 0.5) {
            aucroc = 1 - aucroc;
        }

        try (PrintWriter pw = new PrintWriter(new FileOutputStream(csvFile, true))) {
            pw.write(projectName + "," + modelName + "," + accuracy + "," + precision + "," + recall + "," + fmeasure + "," + aucroc + "\n");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RunExperiment.class.getName()).log(Level.SEVERE, null, ex);
        }

        return fmeasure;
    }

}
