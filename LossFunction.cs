using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PRSGD
{
    /// <summary>
    ///     Interface for a loss function to be used in PR-SGD,
    ///     such a function should have a fixed input dimension d, and we
    ///     must be able to compute the value and he gradient of the function 
    ///     at any vector in R^d
    /// </summary>
    public interface LossFunction
    {
        public float Value(float[] parameters);
        public float[] Gradient(float[] parameters, bool deleteNoise=false);
        public int GetInputDimension();

    }




    /// <summary>
    ///     Q(x1,...,xd) := a1(x1 - c1)^2 + a2(x2 - c2)^2 + ... + ad(xd - cd)^2
    ///      - d is the input dimension
    ///      - a1,..,ad the coefficients
    ///      - c1,..,cd the centers
    /// </summary>
    public class QuadraticLoss: LossFunction
    {
        private float[] _coeffs;
        private float[] _centers;
        private int _inputDimension => _coeffs.Length;
        /// <summary>
        ///     Add noise to the gradients if _noisyGradients == true
        /// </summary>
        private bool _noisyGradients;
        private Random _noiseGenerator = new Random();
            

        public QuadraticLoss(float[] coeffs, float[]? centers = null, bool noisyGrads = true)
        {
            foreach (var coefficient in coeffs)
                if (coefficient < 0)
                    throw new ArgumentException("All coefficients must be non negetive");

            // Set coefficients
            _coeffs = (float[]) coeffs.Clone();

            // Should we add noise to gradients ?
            _noisyGradients = noisyGrads;

            // Set centers
            _centers = SetCenters(centers);
            
        }

        public QuadraticLoss(int inputDimension, float[]? centers = null, bool noisyGrads = true)
        {
            _coeffs = new float[inputDimension];
            for (int i = 0; i < inputDimension; i++)
                _coeffs[i] = 1.0f;

            // Should we add noise to gradients ?
            _noisyGradients = noisyGrads;

            // Set centers
            _centers = SetCenters(centers);
        }


        private float[] SetCenters(float[]? centers = null)
        {
            if (centers == null)
                return new float[_coeffs.Length];
            else
            {
                if (centers.Length != _coeffs.Length)
                    throw new ArgumentException("Must have as many centers as coefficients");
                return (float[])centers.Clone();
            }
        }


        /// <returns>
        ///     a1(x1 - c1)^2 + a2(x2 - c2)^2 + ... + ad(xd - cd)^2,
        ///     - x1,..,xd the parameters vector
        ///     - a1,..,ad the coefficients
        ///     - c1,..,cd the centers
        /// </returns>
        public float Value(float[] parameters)
        {
            if (_coeffs.Length != _inputDimension)
                throw new ArgumentException("Parameters dimension must match the function's input dimension");


            var value = 0.0f;
            for (var i = 0; i < _coeffs.Length; i++)
            {
                var distToCenter = parameters[i] - _centers[i];
                value += _coeffs[i] * distToCenter * distToCenter;
            }

            return value;
        }


        /// <returns>
        ///     the gradient 2 * (x1 - c1, ..., xd - cd) 
        /// </returns>
        public float[] Gradient(float[] parameters, bool deleteNoise=false)
        {
            if (_coeffs.Length != _inputDimension)
                throw new ArgumentException("Parameters dimension must match the function's input dimension");

            var gradient = new float[_inputDimension];
            for (var i = 0; i < _coeffs.Length; i++)
            {
                gradient[i] = 2 * _coeffs[i] * (parameters[i] - _centers[i]);
                if (_noisyGradients && !deleteNoise)
                {
                    gradient[i] += GenerateNoise(parameters[i] - _centers[i]);
                }
            }

            return gradient;
        }


        public int GetInputDimension()
        {
            return _inputDimension;
        }


        public float GenerateNoise(float diff)
        {
            return Math.Max(1, Math.Abs(diff)) * (2 * (float)_noiseGenerator.NextDouble() - 1);
        }
    }








    /// <summary>
    ///     Average of multiple loss functions having the same input dimension
    /// </summary>
    public class AverageLoss: LossFunction
    {
        private LossFunction[] _lossFunctions;
        private int _inputDimension;



        public AverageLoss(LossFunction[] lossFunctions)
        {
            _inputDimension = lossFunctions[0].GetInputDimension();
            // all the loss functions must have the same input dimension
            foreach (var lossFunc in lossFunctions)
                if (lossFunc.GetInputDimension() != _inputDimension)
                    throw new ArgumentException("All the loss functions to average must have the same input dimension");
            
            _lossFunctions = lossFunctions;
        }


        /// <returns>
        ///     average value of all the loss functions
        /// </returns>
        public float Value(float[] parameters)
        {
            float value = 0.0f;
            int countLossFctns = _lossFunctions.Length;

            foreach (var lossFunc in _lossFunctions)
                value += lossFunc.Value(parameters) / countLossFctns;

            return value;
        }


        /// <returns>
        ///     average gradient of all the loss functions
        /// </returns>
        public float[] Gradient(float[] parameters, bool deleteNoise=false)
        {
            var gradient = new float[parameters.Length];
            int countLossFctns = _lossFunctions.Length;

            foreach (var lossFunc in _lossFunctions)
            {
                var currGradient = lossFunc.Gradient(parameters, deleteNoise);
                for (var i = 0; i < gradient.Length; i++)
                {
                    gradient[i] += currGradient[i] / countLossFctns;
                }
            }

            return gradient;
        }


        public int GetInputDimension()
        {
            return _inputDimension;
        }
    }
}






