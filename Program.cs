using System;
using System.Collections.Generic;
namespace NeuralNetwork;

class Program
{
    static void Main(string[] args)
    {
        double[][] inputs = new double[][] { 
        new double[] // 0
        {
            1,1,1,
            1,0,1,
            1,0,1,
            1,0,1,
            1,1,1
        },
        new double[] // 1
        {
            0,1,0,
            1,1,0,
            0,1,0,
            0,1,0,
            1,1,1
        },
        new double[] // 2
        {
            0,1,1,
            1,0,1,
            0,0,1,
            0,1,0,
            1,1,1
        },
        new double[] // 3
        {
            1,1,1,
            0,0,1,
            1,1,1,
            0,0,1,
            1,1,1
        }};
        double[][] targets = new double[][] 
        {
            new double[] { 1, 0, 0, 0 },
            new double[] { 0, 1, 0, 0 },
            new double[] { 0, 0, 1, 0 },
            new double[] { 0, 0, 0, 1 },
        };

        NeuralNetwork neuralNetwork = new NeuralNetwork(15, 2, 4); 
        neuralNetwork.l_rate = 1;
        neuralNetwork.fit(inputs, targets, 100);

        List<double> output;
        double[][] input = new double[][] {
        new double[] // 0
        {
            1,1,1,
            1,0,1,
            1,0,1,
            1,0,1,
            1,1,1
        },
        new double[] // 1
        {
            0,1,0,
            1,1,0,
            0,1,0,
            0,1,0,
            1,1,1
        },
        new double[] // 2
        {
            0,1,1,
            1,0,1,
            0,0,1,
            0,1,0,
            1,1,1
        },
        new double[] // 3
        {
            1,1,1,
            0,0,1,
            1,1,1,
            0,0,1,
            1,1,1
        }};

        foreach (double[] d in input)
        {
            output = neuralNetwork.predict(d);
            foreach (double db in output)
            {
                Console.WriteLine(db);
            }
            Console.WriteLine("");
        }
        
    }
}