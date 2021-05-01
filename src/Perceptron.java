import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner; // Import the Scanner class to read text files


public class Perceptron {

	static ArrayList<double[]> trainingMatrix;
	static ArrayList<String> dictionaryMatrix;
	static ArrayList<double[]> testMatrix;
	public static void main(String []args){
        System.out.println("Hello World");
        Scanner trainingData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project3\\src\\pa3train.txt");
        Scanner testData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project3\\src\\pa3test.txt");
        Scanner dictionaryData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project3\\src\\pa3dictionary.txt");
        trainingMatrix = getMatrixDouble(trainingData);
        testMatrix = getMatrixDouble(testData);
        dictionaryMatrix = getDictionary(dictionaryData);
        ArrayList<Double> labels = new ArrayList<Double>();
        labels.add(1.0);labels.add(2.0);
        trainingMatrix = getRowsWithLabels(labels,trainingMatrix);
        testMatrix = getRowsWithLabels(labels,testMatrix);

        System.out.println("Perceptron Output");
        ArrayList<Double> w1 = perceptron(1,2.0,trainingMatrix);
        System.out.println("1 pass " + "trainingError: " + getVectorError(w1,trainingMatrix) + " test Error: " + getVectorError(w1,testMatrix));
        ArrayList<Double> w2 = perceptron(2,2.0,trainingMatrix);
        System.out.println("2 passes " + "trainingError: " + getVectorError(w2,trainingMatrix) + " test Error: " + getVectorError(w2,testMatrix));
        ArrayList<Double> w3 = perceptron(3,2.0,trainingMatrix);
        System.out.println("3 passes " + "trainingError: " + getVectorError(w3,trainingMatrix) + " test Error: " + getVectorError(w3,testMatrix));
        ArrayList<Double> w4 = perceptron(4,2.0,trainingMatrix);
        System.out.println("4 passes " + "trainingError: " + getVectorError(w4,trainingMatrix) + " test Error: " + getVectorError(w4,testMatrix));
        System.out.println("3 passes top and bottom");
        getTopAndBottom3(w3,dictionaryMatrix);
        System.out.println("Logistic Regression");
        
        ArrayList<Double> l2 = logisticRegression(2,0.001);
        System.out.println("2 iterations " + "trainingError: " + sigmoid(l2,trainingMatrix) + " test Error: " + sigmoid(l2,testMatrix));
        ArrayList<Double> l10 = logisticRegression(10,0.001);
        System.out.println("10 iterations " + "trainingError: " + sigmoid(l10,trainingMatrix) + " test Error: " + sigmoid(l10,testMatrix));
        ArrayList<Double> l50 = logisticRegression(50,0.001);
        System.out.println("50 iterations " + "trainingError: " + sigmoid(l50,trainingMatrix) + " test Error: " + sigmoid(l50,testMatrix));
        ArrayList<Double> l100 = logisticRegression(100,0.001);
        System.out.println("100 iterations " + "trainingError: " + sigmoid(l100,trainingMatrix) + " test Error: " + sigmoid(l100,testMatrix));
        System.out.println("50 iteration top and bottom");
        getTopAndBottom3(l50,dictionaryMatrix);
        
       doConfusion();
        
        
        
	}
	public static void doConfusion() {
		Scanner trainingData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project3\\src\\pa3train.txt");
        trainingMatrix = getMatrixDouble(trainingData);
        Scanner testData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project3\\src\\pa3test.txt");
        testMatrix = getMatrixDouble(testData);
        ArrayList<List<Double>> classifiers = new ArrayList<>();
        for(double i = 1.0; i <= 6.0;i++) {
        	classifiers.add(perceptron(1,i,trainingMatrix));
        }
        
        double[][] confusion = new double[8][7];
        initializeConfusion(confusion);

        for(int i = 1; i <= 6; i++) {
        	for(int j = 1; j <= 6;j++) {
        		confusion[i][j] = multiClassPerceptron((double)i,(double)j,classifiers)/confusion[i][j];	
        	}
        }
        for(int i = 1; i <= 6;i++) {
        	confusion[7][i] = dontKnow(i,classifiers)/confusion[7][i];
        }
        
        System.out.println("----------------------------------------------------------------");
        for(int i = 1; i <= 7; i++) {
        	for(int j = 1; j <= 6;j++) {
        		String s = "" + round(confusion[i][j],3);
        		s = s.length() < 5 ? s +="0" : s;
        		System.out.print((confusion[i][j] == 0.0 ? "  0.0  " : s) + "  | ");
        	}
        	System.out.println("\n----------------------------------------------------------------");
        }
        
	}
	
	public static double round(double value, int places) {
	    if (places < 0) throw new IllegalArgumentException();

	    long factor = (long) Math.pow(10, places);
	    value = value * factor;
	    long tmp = Math.round(value);
	    return (double) tmp / factor;
	}
	
	
	public static double dontKnow( double j, ArrayList<List<Double>>classifiers){
		double total = 0.0;
		for(int k = 0; k < testMatrix.size();k++) {
			if(testMatrix.get(k)[testMatrix.get(0).length - 1] == j) {
				int count = 0;
				for(int n = 0; n < classifiers.size();n++) {
					double prediction = 0.0;
					double y = testMatrix.get(k)[testMatrix.get(0).length - 1] == (double)(n + 1) ? -1.0 : 1.0;
					for(int z = 0; z < classifiers.get(0).size();z++) {
						prediction += (classifiers.get(n).get(z) * testMatrix.get(k)[z]);
					}
					if((prediction * y) >= 0) {
						count += 1;
					}
				}
				if(count != 1) {
					total++;
				}
			}
			
		}
		
		
		return total;
	}
	
	public static double multiClassPerceptron(double i, double j, ArrayList<List<Double>>classifiers){
		double total = 0.0;
		for(int k = 0; k < testMatrix.size();k++) {
			if(testMatrix.get(k)[testMatrix.get(0).length - 1] == j) {
				int count = 0;
				double label = 0;
				for(int n = 0; n < classifiers.size();n++) {
					double prediction = 0.0;
					double y = testMatrix.get(k)[testMatrix.get(0).length - 1] == (double)(n + 1) ? -1.0 : 1.0;
					for(int z = 0; z < classifiers.get(0).size();z++) {
						prediction += (classifiers.get(n).get(z) * testMatrix.get(k)[z]);
					}
					if((prediction * y) >= 0) {
						count += 1;
						if(count > 1) {
							break;
						}
						label = (double)(n + 1);
					}
				}
				if(label == i && count == 1) {
					total += 1;
				}
				
			}
			
			
		}
		
		
		return total;
	}
	
	public static void initializeConfusion(double[][] confusion){
		for(double j = 1.0; j <= 6.0;j++) {
			double num = 0;
			
			//find Nj -- number of test examples that have label j
			for(int i = 0; i < testMatrix.size();i++) {
				if(testMatrix.get(i)[testMatrix.get(0).length - 1] == j) {
					num++;
				}
			}
			for(int i = 1; i <= 7; i++ ) {
				confusion[i][(int)j] = num;
			}
		}
	}
	
	public static void getTopAndBottom3(ArrayList<Double> vector,ArrayList<String> dictionary) {
		PriorityQueue<Double>top3 = new PriorityQueue<Double>();
		PriorityQueue<Double>bottom3 = new PriorityQueue<Double>(Collections.reverseOrder()); 
		for(int i = 0; i < vector.size();i++) {
			top3.add(vector.get(i));
			bottom3.add(vector.get(i));
			if(i > 2) {
				top3.remove();
				bottom3.remove();
			}
		}
		
		System.out.println("Top Values");
		while(!top3.isEmpty()) {
			double d = top3.remove();
			int index = vector.indexOf(d);
			System.out.println(d + "  --  " + dictionary.get(index));
		}
		System.out.println("Bottom Values");
		while(!bottom3.isEmpty()) {
			double d = bottom3.remove();
			int index = vector.indexOf(d);
			System.out.println(d + "  --  " + dictionary.get(index));
		}
		
	}
	
	public static ArrayList<Double> logisticRegression(int iterations,double stepSize) {
		ArrayList<Double> w = new ArrayList<>();
		for(int i = 0; i < trainingMatrix.get(0).length - 1; i++) {
			w.add(0.0);
		}
		
		for(int i = 0; i < iterations;i++) {
			ArrayList<Double> gradientVector = gradientDescent(w,trainingMatrix);
			for(int j = 0;j < gradientVector.size();j++) {
				gradientVector.set(j, gradientVector.get(j) * stepSize);
			}
			for(int j = 0; j < trainingMatrix.get(0).length - 1;j++) {
				w.set(j, w.get(j) + gradientVector.get(j));
			}
		}
		return w;
		
	}
	
	public static ArrayList<Double> gradientDescent(ArrayList<Double> w,ArrayList<double[]> matrix){
		ArrayList<Double> totalVector = new ArrayList<Double>();
		for(int i = 0; i < matrix.size();i++) {
			double y = matrix.get(i)[matrix.get(0).length - 1] != 2.0 ? -1.0 : 1.0;
			ArrayList<Double> yTimesX = new ArrayList<Double>();
			for(int j = 0; j < matrix.get(i).length - 1;j++) {
				yTimesX.add(y * matrix.get(i)[j]);
			}

			double yTimesWTimesX = y * dotProduct(w,matrix.get(i));
			double denominator = 1 + Math.exp(yTimesWTimesX);

			for(int j = 0; j < matrix.get(i).length - 1;j++) {
				yTimesX.set(j,yTimesX.get(j) / denominator);
			}
			if(i == 0) {totalVector = yTimesX;}else {
				for(int j = 0; j < totalVector.size();j++) {
					totalVector.set(j, totalVector.get(j) + yTimesX.get(j));
				}
			}
		}
		return totalVector;
	}
	
	public static double sigmoid(ArrayList<Double> w, ArrayList<double[]> matrix) {
		double error = 0.0;
		for(int i = 0; i < matrix.size();i++) {
			int total = 0;
			double y = matrix.get(i)[matrix.get(0).length - 1] != 2.0 ? -1.0 : 1.0;
			for(int j = 0; j < w.size();j++) {
				total += (-1 * w.get(j) * matrix.get(i)[j]);
			}
			double prediction = 1/(1 + Math.exp( total));
			double pred = prediction >= 0.5 ? 1.0 : -1.0;
			if(pred != y) {
				error++;
			}
			
		}
		return error/matrix.size();
	}
	
	public static double getVectorError(ArrayList<Double> w, ArrayList<double[]> matrix) {
		double error = 0.0;
		for(int i = 0; i < matrix.size();i++) {
			int total = 0;
			double y = matrix.get(i)[matrix.get(0).length - 1] != 2.0 ? -1.0 : 1.0;
			for(int j = 0; j < w.size();j++) {
				total += (w.get(j) * matrix.get(i)[j]);
			}
			if((total * y) < 0) {
				error += 1;
			}
			if((total * y) == 0) {
				if(Math.random() < 0.5) {
					error += 1;
				}
			}
			
		}
		return error/matrix.size();
	}
	
	public static ArrayList<Double> perceptron(int numPasses,double oneVsAll,ArrayList<double[]> trainingMatrix){

		ArrayList<Double> w = new ArrayList<>();
		for(int i = 0; i < trainingMatrix.get(0).length - 1; i++) {
			w.add(0.0);
		}
		for(int k = 0; k < numPasses;k++) {
			for(int i = 0; i < trainingMatrix.size();i++) {

				double y = trainingMatrix.get(i)[trainingMatrix.get(0).length - 1] != oneVsAll ? -1.0 : 1.0;
				double wTimesX = dotProduct(w,trainingMatrix.get(i)); 
				if((y * wTimesX) <= 0 ) {
					ArrayList<Double> yTimesX = new ArrayList<Double>();
					for(int j = 0; j < trainingMatrix.get(i).length - 1;j++) {
						yTimesX.add(y * trainingMatrix.get(i)[j]);
					}
					for(int j = 0; j < w.size();j++) {
						w.set(j, w.get(j) + yTimesX.get(j));
					}
				}
			}
		}

		return w;
	}
	
	public static double dotProduct(ArrayList<Double> first,double[] second) {
		double total = 0.0;
		int size = first.size() < second.length ? first.size() : second.length;
		for(int i = 0; i < size;i++) {
			total += (first.get(i) * second[i]);
		}
		return total;
	}
	
	
	
	public static ArrayList<double[]> getRowsWithLabels(ArrayList<Double> labels, ArrayList<double[]> matrix){
		for(int i = 0; i < matrix.size();i++) {
			if(!labels.contains(matrix.get(i)[matrix.get(0).length - 1])) {
				matrix.remove(i);
				i--;
			}
		}
		return matrix;
	}
	
	public static ArrayList<String> getDictionary(Scanner data){
		ArrayList<String> matrix = new ArrayList<>();
		while(data.hasNextLine()) {
			String dataString = data.nextLine();
			matrix.add(dataString);
			
		}
		return matrix;
	}
	
 	public static ArrayList<double[]> getMatrixDouble(Scanner data) {
		ArrayList<double[]> matrix = new ArrayList<>();
		while(data.hasNextLine()) {
			String dataString = data.nextLine();
			String[] dataArray = dataString.split(" ");
			double[] n1 = new double[dataArray.length];
			
			for(int i = 0; i < dataArray.length; i++) {
			   n1[i] = Double.parseDouble(dataArray[i]);
			}
			matrix.add(n1);
			
		}
		return matrix;
	}
	
	public static ArrayList<int[]> getMatrixInt(Scanner data) {
		ArrayList<int[]> matrix = new ArrayList<>();
		while(data.hasNextLine()) {
			String dataString = data.nextLine();
			String[] dataArray = dataString.split(" ");
			int[] n1 = new int[dataArray.length];
			for(int i = 0; i < dataArray.length; i++) {
			   n1[i] = Integer.parseInt(dataArray[i]);
			}
			matrix.add(n1);
			
		}
		return matrix;
	}
	
	
	public static Scanner readFile(String fileName) {
		try {
		      File myObj = new File(fileName);
		      Scanner myReader = new Scanner(myObj);
		      return myReader;
		}
		catch(FileNotFoundException e) {
			System.out.println("An error occured.");
			e.printStackTrace();
			return null;
		}
	}
	
	
}
/*
 * import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Scanner; // Import the Scanner class to read text files


public class Boosting {

	static ArrayList<double[]> trainingMatrix;
	static ArrayList<String> dictionaryMatrix;
	static ArrayList<double[]> testMatrix;
	static ArrayList<Double> aList = new ArrayList<>();
	static ArrayList<Integer> hClassifiers = new ArrayList<>();
	static ArrayList<Integer> dictionaryHIndices = new ArrayList<>();;
	public static void main(String []args){
        System.out.println("Hello World");
        Scanner trainingData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project4\\src\\pa5train.txt");
        Scanner testData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project4\\src\\pa5test.txt");
        Scanner dictionaryData = readFile("C:\\Users\\Michael\\eclipse-workspace\\CSE 151A Project4\\src\\pa5dictionary.txt");
        trainingMatrix = getMatrixDouble(trainingData);
        testMatrix = getMatrixDouble(testData);
        dictionaryMatrix = getDictionary(dictionaryData);
        ArrayList<Integer> iterations = new ArrayList<Integer>();
        //iterations.add(3);
        iterations.add(4);
        //iterations.add(7);iterations.add(10);iterations.add(15);iterations.add(20);
        //example
        trainingMatrix = new ArrayList<>();
        trainingMatrix.add(new double[]{1,1,1});
        trainingMatrix.add(new double[]{2,1,-1});
        trainingMatrix.add(new double[]{4,1,-1});
        trainingMatrix.add(new double[]{1,2,1});
        trainingMatrix.add(new double[]{2,2,-1});
        trainingMatrix.add(new double[]{3,2,-1});
        trainingMatrix.add(new double[]{2,3,1});
        trainingMatrix.add(new double[]{2,4,1});
        trainingMatrix.add(new double[]{3,3,1});
        trainingMatrix.add(new double[]{4,3,-1});
        for(int i = 0; i < iterations.size();i++) {
        	boosting(3);
        	System.out.println("iteration " + iterations.get(i) + " training E: " + getError(trainingMatrix) + " test E: " + getError(testMatrix));
        	if(iterations.get(i) == 10) {
        		getWords();
        	}
        }
        
        
	}
	
	public static void getWords() {
		System.out.println("WORDS");
		for(int i = 0; i < dictionaryHIndices.size();i++) {
			System.out.println(dictionaryMatrix.get(dictionaryHIndices.get(i)));
		}
	}
	
	public static double getError(ArrayList<double[]> matrix) {
		if(aList.size() != hClassifiers.size() || aList.size() != dictionaryHIndices.size()) {
			System.out.println("ERROR");
			return 0;
		}
		double errorCount = 0;
		for(int i = 0; i < matrix.size(); i++) {
			double sign = 0;
			for(int j = 0; j < aList.size(); j++) {
				sign += (aList.size() * getHLabel(hClassifiers.get(j),dictionaryHIndices.get(j),matrix.get(i)));
			}
			double actualLabel = matrix.get(i)[matrix.get(i).length - 1];
			if((sign > 0.0 && 1.0 != actualLabel) || (sign < 0.0 && -1.0 != actualLabel) ) {
				errorCount++;
			}
		}
		return errorCount / matrix.size();
	}
	
	public static void boosting(int iterations) {
		//hPlus == 1 
		aList = new ArrayList<>();
		hClassifiers = new ArrayList<>();
		dictionaryHIndices = new ArrayList<>();
		ArrayList<Double> weights = new ArrayList<>();
		for(int i = 0; i < trainingMatrix.size();i++) {
			weights.add(1.0/(double)trainingMatrix.size());
		}
		for(int i = 0; i < iterations;i++) {
			//double Et = getWeakLearner(weights);
			double Et = getWeakLearnerExample(weights);
			int h = Et >= 0 ? 1 : -1;
			System.out.println(Et);
			Et = h * Et;
			double temp = Math.floor(Et);
			int dictionaryIndex = (int)temp;
			Et = Et - temp;
			System.out.println(Et);
			System.out.println("dict " + dictionaryIndex);
			Et = Double.parseDouble(String.format("%.2f", Et));
			System.out.println(Et);
			double hi = (1-Et)/Et;
			System.out.println("og " + hi);
			
			double At = 0.5 * Math.log((1-Et)/Et);//closer to 0, bigger At is 
			double Z = 0.0;
			for(int j = 0; j < weights.size();j++) {
				int hLabel = getHLabel(h,dictionaryIndex,trainingMatrix.get(j));
				System.out.println("[ " + trainingMatrix.get(j)[0]  + ", " + trainingMatrix.get(j)[1] +", "+ trainingMatrix.get(j)[2] + "]");
				System.out.println(trainingMatrix.get(j)[trainingMatrix.get(0).length - 1]  + " hLabel " + hLabel);
				System.out.println("wT " + weights.get(j));
				System.out.println("ePower " + (-1 * At * trainingMatrix.get(j)[trainingMatrix.get(0).length - 1] * hLabel ));
				double newWeight = weights.get(j) * Math.exp(-1 * At * trainingMatrix.get(j)[trainingMatrix.get(0).length - 1] * hLabel );
				System.out.println(newWeight);
				weights.set(j, newWeight);
				Z += newWeight;
			}
			System.out.println("Z "+ Z);
			for(int j = 0; j < weights.size();j++) {
				weights.set(j, weights.get(j) / Z);
			}
			System.out.println("At " + At);
			for(int j = 0; j < weights.size();j++) {
				System.out.print(weights.get(j) + " ");
			}
			aList.add(At);
			hClassifiers.add(h);
			dictionaryHIndices.add(dictionaryIndex);
							
		}
		
		
		
	}
	
	public static int getHLabel(int h,int dictionaryIndex,double[] row) {
//		if(h == 1) {
//			return row[dictionaryIndex] == 1.0 ? 1 : -1;
//		}else {
//			return row[dictionaryIndex] == 0.0 ? 1 : -1;
//		}
		if(h == 1) {
			if(row[0] <= dictionaryIndex + 0.5) {
				return 1;
			}else {
				return -1;
			}
		}else {
			if(row[1] > dictionaryIndex + 0.5) {
				return 1;
			}else {
				return -1;
			}
		}		
	}
	public static double getWeakLearnerExample(ArrayList<Double> weights) {
		double leastError = 1.0;
		int dictionaryIndex = 0;
		int hPlus = 1;
		for(int i = 1; i < 4;i++) {
			double line = i + 0.5;
			ArrayList<Double> h1PlusOutput = new ArrayList<>();
			ArrayList<Double> h1MinusOutput = new ArrayList<>();
			for(int j = 0; j < trainingMatrix.size() ;j++) {
				if(trainingMatrix.get(j)[0] <= line) {
					h1PlusOutput.add(1.0);
				}else {
					h1PlusOutput.add(-1.0);
				}
				if(trainingMatrix.get(j)[1] > line) {
					h1MinusOutput.add(1.0);
				}else {
					h1MinusOutput.add(-1.0);
				}
			}

			
			//error_w(h+)
			double errorHPlus = 0.0,errorHMinus = 0.0;
			for(int k = 0; k < h1PlusOutput.size();k++) {
			
				double label = trainingMatrix.get(k)[trainingMatrix.get(0).length - 1];
				errorHPlus += weights.get(k) * (h1PlusOutput.get(k) != label ? 1.0 : 0.0);
				errorHMinus += weights.get(k) * (h1MinusOutput.get(k) != label ? 1.0 : 0.0);
			}
//			System.out.println("errorHPlus " + errorHPlus + "n");
//			System.out.println("errorHMinus " + errorHMinus + "n");
			if(leastError >= errorHPlus || leastError >= errorHMinus) {
				dictionaryIndex = i;
				leastError = Math.min(errorHMinus, errorHPlus);
				hPlus = leastError == errorHPlus ? 1 : -1;
			}
		}
		System.out.println("least " + leastError);
		return hPlus * (dictionaryIndex + leastError);

	}
	
	public static double getWeakLearner(ArrayList<Double> weights) {
		double leastError = 1.0;
		int dictionaryIndex = 0;
		int hPlus = 1;
		for(int i = 0; i < trainingMatrix.get(0).length - 1;i++) {
			ArrayList<Double> h1PlusOutput = new ArrayList<>();
			ArrayList<Double> h1MinusOutput = new ArrayList<>();
			for(int j = 0; j < trainingMatrix.size() ;j++) {
				if(trainingMatrix.get(j)[i] == 1.0) {
					h1PlusOutput.add(1.0);
					h1MinusOutput.add(-1.0);
				}else {
					h1PlusOutput.add(-1.0);
					h1MinusOutput.add(1.0);
				}
			}

			
			//error_w(h+)
			double errorHPlus = 0.0,errorHMinus = 0.0;
			for(int k = 0; k < h1PlusOutput.size();k++) {
				double label = trainingMatrix.get(k)[trainingMatrix.get(0).length - 1];
				errorHPlus += weights.get(k) * (h1PlusOutput.get(k) != label ? 1.0 : 0.0);
				errorHMinus += weights.get(k) * (h1MinusOutput.get(k) != label ? 1.0 : 0.0);
			}
			System.out.println("errorHPlus " + errorHPlus + "n");
			System.out.println("errorHMinus " + errorHMinus + "n");
			if(leastError >= errorHPlus || leastError >= errorHMinus) {
				dictionaryIndex = i;
				leastError = Math.min(errorHMinus, errorHPlus);
				hPlus = leastError == errorHPlus ? 1 : -1;
			}
		}
		System.out.println("least " + leastError);
		return hPlus * (dictionaryIndex + leastError);
	}
	
	public static ArrayList<String> getDictionary(Scanner data){
		ArrayList<String> matrix = new ArrayList<>();
		while(data.hasNextLine()) {
			String dataString = data.nextLine();
			matrix.add(dataString);
			
		}
		return matrix;
	}
	
 	public static ArrayList<double[]> getMatrixDouble(Scanner data) {
		ArrayList<double[]> matrix = new ArrayList<>();
		while(data.hasNextLine()) {
			String dataString = data.nextLine();
			String[] dataArray = dataString.split(" ");
			double[] n1 = new double[dataArray.length];
			
			for(int i = 0; i < dataArray.length; i++) {
			   n1[i] = Double.parseDouble(dataArray[i]);
			}
			matrix.add(n1);
			
		}
		return matrix;
	}
	
	public static ArrayList<int[]> getMatrixInt(Scanner data) {
		ArrayList<int[]> matrix = new ArrayList<>();
		while(data.hasNextLine()) {
			String dataString = data.nextLine();
			String[] dataArray = dataString.split(" ");
			int[] n1 = new int[dataArray.length];
			for(int i = 0; i < dataArray.length; i++) {
			   n1[i] = Integer.parseInt(dataArray[i]);
			}
			matrix.add(n1);
			
		}
		return matrix;
	}
	
	
	public static Scanner readFile(String fileName) {
		try {
		      File myObj = new File(fileName);
		      Scanner myReader = new Scanner(myObj);
		      return myReader;
		}
		catch(FileNotFoundException e) {
			System.out.println("An error occured.");
			e.printStackTrace();
			return null;
		}
	}
}*/
 */

