using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PRSGD
{
    internal class Worker
    {
        public int _workerId;
        private LossFunction _lossFunction;
        private float[] _localParameters;


        public Worker(int id, LossFunction lossFunction)
        {
            _workerId = id;
            _lossFunction = lossFunction;
            _localParameters = new float[lossFunction.GetInputDimension()];
        }


        public void SetParameters(float[] parameters)
        {
            for (var i = 0; i < parameters.Length; i++)
                _localParameters[i] = parameters[i];
        }

        public float[] GetParameters()
        {
            return _localParameters;
        }


        /// <summary>
        ///     runs a local SGD optimizing the _lossFunction and updating _localParameters
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="numSteps"></param>
        public void LocalSGD(float learningRate, int numSteps)
        {
 
            for (int step = 0; step < numSteps; step++)
            {
                // one step of SGD
                var gradient = _lossFunction.Gradient(_localParameters);
                for (var i = 0; i < gradient.Length; i++)
                {
                    _localParameters[i] -= learningRate * gradient[i];
                }
            }
        }

    }
}
