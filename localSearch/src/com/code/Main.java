package com.code;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class Main {

    public static void main(String[] args) throws IOException {
        String path = "../../../data/";
        runGreedy2nRExperiment(path);
    }

    static double[][] getDistMatrix(String path) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(path));
        int size = lines.size();
        double[][] coords = new double[size][2];
        double[] costs = new double[size];
        for (int i = 0; i < size; i++) {
            String[] parts = lines.get(i).split(";");
            coords[i][0] = Double.parseDouble(parts[0]);
            coords[i][1] = Double.parseDouble(parts[1]);
            costs[i] = Double.parseDouble(parts[2]);
        }
        double[][] matrix = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    matrix[i][j] = Double.POSITIVE_INFINITY;
                } else {
                    double dx = coords[i][0] - coords[j][0];
                    double dy = coords[i][1] - coords[j][1];
                    matrix[i][j] = Math.hypot(dx, dy) + costs[i];
                }
            }
        }
        return matrix;
    }

    static double calculateCost(int[] solution, double[][] matrix) {
        double cost = 0;
        for (int i = 1; i < solution.length; i++) {
            cost += matrix[solution[i - 1]][solution[i]];
        }
        return cost;
    }

    static double[] findRegretWithSolution(List<Integer> solution, int vertexId, double[][] matrix) {
        List<Double> costs = new ArrayList<>();
        List<List<Integer>> solutions = new ArrayList<>();
        for (int i = 0; i <= solution.size(); i++) {
            List<Integer> newSol = new ArrayList<>(solution);
            newSol.add(i, vertexId);
            solutions.add(newSol);
            costs.add(calculateCost(newSol, matrix));
        }
        return getRegretNSol(costs, solutions);
    }

    static double[] getRegretNSol(List<Double> costs, List<List<Integer>> solutions) {
        int first = IntStream.range(0, costs.size()).boxed().min(Comparator.comparing(costs::get)).orElse(-1);
        double cost1 = costs.get(first);
        List<Integer> sol = solutions.get(first);
        costs.remove(first);
        solutions.remove(first);
        int second = IntStream.range(0, costs.size()).boxed().min(Comparator.comparing(costs::get)).orElse(-1);
        double cost2 = costs.get(second);
        return new double[]{cost2 - cost1, sol.stream().mapToDouble(i -> i).toArray()};
    }

    static double[] weightedRegret(double[][] matrix, int startV, double weight) {
        int n = (int) Math.ceil(matrix.length / 2.0);
        int nextV = IntStream.range(0, matrix[startV].length).boxed().min(Comparator.comparingDouble(i -> matrix[startV][i])).orElse(-1);
        List<Integer> cycle = new ArrayList<>(Arrays.asList(startV, nextV));
        double currentCost = calculateCost(cycle, matrix);
        boolean[] unvisited = new boolean[matrix.length];
        Arrays.fill(unvisited, true);
        unvisited[startV] = false;
        unvisited[nextV] = false;
        for (int i = 0; i < n - 2; i++) {
            double[] scores = new double[unvisited.length];
            Arrays.fill(scores, Double.NEGATIVE_INFINITY);
            double[] newCosts = new double[unvisited.length];
            List<Integer>[] newSols = new List[unvisited.length];
            for (int vertexId = 0; vertexId < unvisited.length; vertexId++) {
                if (unvisited[vertexId]) {
                    double[] regretNSol = findRegretWithSolution(cycle, vertexId, matrix);
                    double regret = regretNSol[0];
                    List<Integer> solution = Arrays.stream(regretNSol[1]).boxed().collect(Collectors.toList());
                    double newCost = calculateCost(solution, matrix);
                    double increase = newCost - currentCost;
                    double score = weight * regret - (1 - weight) * increase;
                    scores[vertexId] = score;
                    newSols[vertexId] = solution;
                    newCosts[vertexId] = newCost;
                }
            }
            int highestScoreId = IntStream.range(0, scores.length).boxed().max(Comparator.comparingDouble(i -> scores[i])).orElse(-1);
            cycle = newSols[highestScoreId];
            unvisited[highestScoreId] = false;
            currentCost = newCosts[highestScoreId];
        }
        return new double[]{cycle.stream().mapToDouble(i -> i).toArray(), currentCost};
    }

    static double[] randomSolution(double[][] matrix) {
        int n = (int) Math.ceil(matrix.length / 2.0);
        List<Integer> sol = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int randomIndex;
            do {
                randomIndex = (int) (Math.random() * matrix.length);
            } while (sol.contains(randomIndex));
            sol.add(randomIndex);
        }
        double cost = calculateCost(sol, matrix);
        return new double[]{sol.stream().mapToDouble(i -> i).toArray(), cost};
    }

    static double calculateDelta(List<Integer> solution, double[][] matrix, int mIdIn, int sIdOut) {
        int prevVertex = solution.get((sIdOut - 1 + solution.size()) % solution.size());
        int nextVertex = solution.get((sIdOut + 1) % solution.size());
        double costOut = matrix[prevVertex][solution.get(sIdOut)] + matrix[solution.get(sIdOut)][nextVertex];
        double costIn = matrix[prevVertex][mIdIn] + matrix[mIdIn][nextVertex];
        return costIn - costOut;
    }

    static List<double[]> getNeighbourhood2n(List<Integer> solution, double[][] matrix, int numReplacements) {
        List<double[]> neighbors = new ArrayList<>();
        // intra-route
        for (int i = 0; i < solution.size() - 1; i++) {
            for (int j = i + 1; j < solution.size(); j++) {
                List<Integer> neighbor = new ArrayList<>(solution);
                Collections.swap(neighbor, i, j);
                double delta = calculateDelta(solution, matrix, neighbor.get(i), j);
                neighbors.add(new double[]{neighbor.stream().mapToDouble(k -> k).toArray(), delta});
            }
        }
        Set<Integer> allNodes = IntStream.range(0, matrix.length).boxed().collect(Collectors.toSet());
        List<Integer> availableNodes = new ArrayList<>(allNodes);
        availableNodes.removeAll(solution);
        // inter-route
        for (int i = 0; i < solution.size(); i++) {
            for (int j = 0; j < availableNodes.size(); j++) {
                List<Integer> neighbor = new ArrayList<>(solution);
                neighbor.set(i, availableNodes.get(j));
                double delta = calculateDelta(solution, matrix, neighbor.get(i), j);
                neighbors.add(new double[]{neighbor.stream().mapToDouble(k -> k).toArray(), delta});
            }
        }
        return neighbors;
    }

    static double[] steepest2n(double[][] matrix, List<Integer> startingSol) {
        List<Integer> bestSol = startingSol;
        double bestDelta = 0;
        List<double[]> neighbourhood = getNeighbourhood2n(startingSol, matrix, 40);
        while (!neighbourhood.isEmpty()) {
            double[] bestNeighbor = neighbourhood.stream().min(Comparator.comparingDouble(n -> n[1])).orElse(null);
            if (bestNeighbor == null || bestNeighbor[1] >= 0) {
                break;
            }
            bestSol = Arrays.stream(bestNeighbor[0]).boxed().collect(Collectors.toList());
            bestDelta = bestNeighbor[1];
            neighbourhood = getNeighbourhood2n(bestSol, matrix, 40);
        }
        return new double[]{bestSol.stream().mapToDouble(i -> i).toArray(), calculateCost(bestSol, matrix)};
    }

    static double[] getRandomNeighbour2n(List<Integer> solution, double[][] matrix) {
        List<Integer> neighbor = new ArrayList<>(solution);
        Set<Integer> allNodes = IntStream.range(0, matrix.length).boxed().collect(Collectors.toSet());
        List<Integer> availableNodes = new ArrayList<>(allNodes);
        availableNodes.removeAll(solution);
        String action = Math.random() < 0.5 ? "swap" : "replace";
        int i, j;
        double delta;
        if ("swap".equals(action)) {
            i = (int) (Math.random() * solution.size());
            j = (int) (Math.random() * solution.size());
            Collections.swap(neighbor, i, j);
            delta = calculateDelta(solution, matrix, neighbor.get(i), j);
        } else {
            i = (int) (Math.random() * solution.size());
            j = (int) (Math.random() * availableNodes.size());
            neighbor.set(i, availableNodes.get(j));
            delta = calculateDelta(solution, matrix, availableNodes.get(j), i);
        }
        return new double[]{neighbor.stream().mapToDouble(k -> k).toArray(), delta};
    }

    static double[] greedy2n(double[][] matrix, List<Integer> startingSol) {
        List<Integer> bestSol = startingSol;
        double[] probablyBestSol = getRandomNeighbour2n(startingSol, matrix);
        while (probablyBestSol[1] < 0) {
            bestSol = Arrays.stream(probablyBestSol[0]).boxed().collect(Collectors.toList());
            probablyBestSol = getRandomNeighbour2n(bestSol, matrix);
        }
        return new double[]{bestSol.stream().mapToDouble(i -> i).toArray(), calculateCost(bestSol, matrix)};
    }

    public static void runGreedy2nRExperiment(String path) throws IOException {
        double[][] matrix = getDistMatrix(path);
        List<double[]> solutions = new ArrayList<>();
        for (int v = 0; v < 200; v++) {
            solutions.add(greedy2n(matrix, randomSolution(matrix)));
        }
        double[] costs = solutions.stream().mapToDouble(sol -> sol[1]).toArray();
        double[] bestSol = solutions.stream().min(Comparator.comparingDouble(sol -> sol[1])).orElse(null);
        double[] worstSol = solutions.stream().max(Comparator.comparingDouble(sol -> sol[1])).orElse(null);
        double avgCost = Arrays.stream(costs).average().orElse(Double.NaN);
        System.out.println("Best cost: " + bestSol[1]);
        System.out.println("Worst cost: " + worstSol[1]);
        System.out.println("Mean cost after 200 solutions: " + avgCost);
        writeSolutionToFile("best_tour.txt", bestSol);
        writeSolutionToFile("worst_tour.txt", worstSol);
    }

    static void writeSolutionToFile(String filename, double[] solution) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("Solution: " + Arrays.toString(solution[0]));
            writer.println("Cost: " + solution[1]);
        }
    }
}

