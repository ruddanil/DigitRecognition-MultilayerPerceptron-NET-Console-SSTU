using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Matrix
    {
        Random rnd = new Random();
        double[][] data;
        int rows, cols;

        public Matrix(int rows, int cols)
        {
            data = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                data[i] = new double[cols];
            }
            this.rows = rows;
            this.cols = cols;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    data[i][j] = (new Random().NextDouble() * 2) - 1;
                }
            }
        }
        public void add(double scaler)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i][j] += scaler;
                }
            }
        }
        public void add(Matrix m)
        {
            if (cols != m.cols || rows != m.rows)
            {
                Console.WriteLine("Shape Mismatch");
                return;
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i][j] += m.data[i][j];
                }
            }
        }
        public static Matrix subtract(Matrix a, Matrix b)
        {
            Matrix temp = new Matrix(a.rows, a.cols);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.cols; j++)
                {
                    temp.data[i][j] = a.data[i][j] - b.data[i][j];
                }
            }
            return temp;
        }
        public static Matrix transpose(Matrix a)
        {
            Matrix temp = new Matrix(a.cols, a.rows);
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.cols; j++)
                {
                    temp.data[j][i] = a.data[i][j];
                }
            }
            return temp;
        }
        public void sigmoid()
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i][j] = 1 / (1 + Math.Exp(-this.data[i][j]));
                }
            }
        }
        public Matrix dsigmoid()
        {
            Matrix temp = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    temp.data[i][j] = this.data[i][j] * (1 - this.data[i][j]);
                }
            }
            return temp;
        }
        public static Matrix multiply(Matrix a, Matrix b)
        {
            Matrix temp = new Matrix(a.rows, b.cols);
            for (int i = 0; i < temp.rows; i++)
            {
                for (int j = 0; j < temp.cols; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < a.cols; k++)
                    {
                        sum += a.data[i][k] * b.data[k][j];
                    }
                    temp.data[i][j] = sum;
                }
            }
            return temp;
        }
        public void multiply(Matrix a)
        {
            for (int i = 0; i < a.rows; i++)
            {
                for (int j = 0; j < a.cols; j++)
                {
                    this.data[i][j] *= a.data[i][j];
                }
            }

        }
        public void multiply(double a)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i][j] *= a;
                }
            }

        }
        public static Matrix fromArray(double[] x)
        {
            Matrix temp = new Matrix(x.Length, 1);

            for (int i = 0; i < x.Length; i++)
            {
                temp.data[i][0] = x[i];
            }    
                
            return temp;
        }
        public List<double> toArray()
        {
            List<double> temp = new List<double>();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    temp.Add(data[i][j]);
                }
            }

            return temp;
        }
        public static double mseLoss(Matrix m)
        {
            double[] diff = new double[m.rows];
            double sum_sq = 0;
            for (int i = 0; i < m.rows; i++)
            {
                sum_sq += Math.Pow(m.data[m.rows - 1][0], 2);
            }

            return sum_sq / m.rows;
        }

    }
}
