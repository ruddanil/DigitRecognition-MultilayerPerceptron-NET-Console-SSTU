namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly Matrix _weightsIh;
        private readonly Matrix _weightsHo;
        private readonly Matrix _biasH;
        private readonly Matrix _biasO;
        public double LRate = 0.01; // Learning Rate
        private double[] _lossArr;

        public NeuralNetwork(int i, int h, int o)
        {
            _weightsIh = new Matrix(h, i); // Matrix of weights for the input and hidden layer
            _weightsHo = new Matrix(o, h); // Matrix of weights for the hidden and output layer

            _biasH = new Matrix(h, 1); // Offset matrix for the hidden layer
            _biasO = new Matrix(o, 1); // Offset matrix for the output layer
        }
        public List<double> Predict(double[] x)
        {
            Matrix input = Matrix.FromArray(x);
            Matrix hidden = Matrix.Multiply(_weightsIh, input);
            hidden.Add(_biasH);
            hidden.Sigmoid();

            Matrix output = Matrix.Multiply(_weightsHo, hidden);
            output.Add(_biasO);
            output.Sigmoid();

            return output.ToArray();
        }
        public void Fit(double[][] x, double[][] y, int epochs)
        {
            _lossArr = new double[epochs];
            Random rnd = new();
            for (int i = 0; i < epochs; i++)
            {
                int sampleN = rnd.Next(x.Length);
                Train(x[sampleN], y[sampleN]);

                Matrix input = Matrix.FromArray(x[sampleN]);
                Matrix hidden = Matrix.Multiply(_weightsIh, input);
                hidden.Add(_biasH);
                hidden.Sigmoid();

                Matrix output = Matrix.Multiply(_weightsHo, hidden);
                output.Add(_biasO);
                output.Sigmoid();

                Matrix target = Matrix.FromArray(y[sampleN]);

                Matrix error = Matrix.Subtract(target, output);
                _lossArr[i] = Matrix.MseLoss(error);
            }
        }
        public void Train(double[] x, double[] y)
        {
            Matrix input = Matrix.FromArray(x);
            Matrix hidden = Matrix.Multiply(_weightsIh, input);
            hidden.Add(_biasH);
            hidden.Sigmoid();

            Matrix output = Matrix.Multiply(_weightsHo, hidden);
            output.Add(_biasO);
            output.Sigmoid();

            Matrix target = Matrix.FromArray(y);

            Matrix error = Matrix.Subtract(target, output);
            Matrix gradient = output.Dsigmoid();
            gradient.Multiply(error);
            gradient.Multiply(LRate);

            Matrix hiddenT = Matrix.Transpose(hidden);
            Matrix whoDelta = Matrix.Multiply(gradient, hiddenT);

            _weightsHo.Add(whoDelta);
            _biasO.Add(gradient);

            Matrix whoT = Matrix.Transpose(_weightsHo);
            Matrix hiddenErrors = Matrix.Multiply(whoT, error);

            Matrix hGradient = hidden.Dsigmoid();
            hGradient.Multiply(hiddenErrors);
            hGradient.Multiply(LRate);

            Matrix iT = Matrix.Transpose(input);
            Matrix wihDelta = Matrix.Multiply(hGradient, iT);

            _weightsIh.Add(wihDelta);
            _biasH.Add(hGradient);
        }
    }
}
