namespace NeuralNetwork
{
    internal class Matrix
    {
        private readonly double[][] _data;
        private readonly int _rows, _cols;

        public Matrix(int rows, int cols)
        {
            _data = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                _data[i] = new double[cols];
            }
            this._rows = rows;
            this._cols = cols;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _data[i][j] = (new Random().NextDouble() * 2) - 1;
                }
            }
        }

        public void Add(Matrix m)
        {
            if (_cols != m._cols || _rows != m._rows)
            {
                Console.WriteLine("Shape Mismatch");
                return;
            }

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    this._data[i][j] += m._data[i][j];
                }
            }
        }
        public static Matrix Subtract(Matrix a, Matrix b)
        {
            Matrix temp = new(a._rows, a._cols);
            for (int i = 0; i < a._rows; i++)
            {
                for (int j = 0; j < a._cols; j++)
                {
                    temp._data[i][j] = a._data[i][j] - b._data[i][j];
                }
            }
            return temp;
        }
        public static Matrix Transpose(Matrix a)
        {
            Matrix temp = new(a._cols, a._rows);
            for (int i = 0; i < a._rows; i++)
            {
                for (int j = 0; j < a._cols; j++)
                {
                    temp._data[j][i] = a._data[i][j];
                }
            }
            return temp;
        }
        public void Sigmoid()
        {
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    this._data[i][j] = 1 / (1 + Math.Exp(-this._data[i][j]));
                }
            }
        }
        public Matrix Dsigmoid()
        {
            Matrix temp = new(_rows, _cols);
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    temp._data[i][j] = this._data[i][j] * (1 - this._data[i][j]);
                }
            }
            return temp;
        }
        public static Matrix Multiply(Matrix a, Matrix b)
        {
            Matrix temp = new(a._rows, b._cols);
            for (int i = 0; i < temp._rows; i++)
            {
                for (int j = 0; j < temp._cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < a._cols; k++)
                    {
                        sum += a._data[i][k] * b._data[k][j];
                    }
                    temp._data[i][j] = sum;
                }
            }
            return temp;
        }
        public void Multiply(Matrix a)
        {
            for (int i = 0; i < a._rows; i++)
            {
                for (int j = 0; j < a._cols; j++)
                {
                    this._data[i][j] *= a._data[i][j];
                }
            }
        }
        public void Multiply(double a)
        {
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    this._data[i][j] *= a;
                }
            }
        }
        public static Matrix FromArray(double[] x)
        {
            Matrix temp = new(x.Length, 1);

            for (int i = 0; i < x.Length; i++)
            {
                temp._data[i][0] = x[i];
            }    
                
            return temp;
        }
        public List<double> ToArray()
        {
            List<double> temp = new();

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    temp.Add(_data[i][j]);
                }
            }

            return temp;
        }
        public static double MseLoss(Matrix m)
        {
            double sumSq = 0;
            for (int i = 0; i < m._rows; i++)
            {
                sumSq += Math.Pow(m._data[m._rows - 1][0], 2);
            }

            return sumSq / m._rows;
        }

    }
}
