namespace NeuralNetwork;

internal class Program
{
    private static void Main()
    {
        // Input data for training (digits from 0 to 3)
        double[][] inputs = { 
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

        // Target values for model training
        double[][] targets = {
            new double[] { 1, 0, 0, 0 },
            new double[] { 0, 1, 0, 0 },
            new double[] { 0, 0, 1, 0 },
            new double[] { 0, 0, 0, 1 },
        };

        NeuralNetwork neuralNetwork = new(15, 2, 4)
        {
            LRate = 1
        };
        neuralNetwork.Fit(inputs, targets, 10000);

        // Input data for recognition
        double[][] input = {
        new double[] // 1
        {
            0,1,0,
            1,1,0,
            0,1,0,
            0,1,0,
            1,1,1
        },
        new double[] // 0
        {
            1,1,1,
            1,0,1,
            1,0,1,
            1,0,1,
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
        },
        new double[] // 3
        {
            1,1,1,
            0,0,1,
            1,1,1,
            0,0,1,
            1,1,1
        }};

        foreach (double[] t in input)
        {
            double[] output = neuralNetwork.Predict(t).ToArray();
            int maxIndex = Array.IndexOf(output, output.Max());
            string maxValue = (output[maxIndex] * 100).ToString("0.##") + "%";
            Console.WriteLine("Digit recognized: " + maxIndex + ", confidence: " + maxValue);
        }
    }
}