using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PRSGD
{
    public class Core
    {
        private Worker[] _workers;
        private int _numWorkers => _workers.Length;
        /// <summary>
        ///     The objective function is the average of the objective functions of
        ///     all the workers, but it is not necessary an instance of AverageLoss, 
        ///     all the the workers might have the same objective function, in that case
        ///     the objective function of the core is the same (no need to averaging)
        /// </summary>
        private LossFunction _averageLoss;

        /// <summary>
        ///     The central parameters array
        /// </summary>
        private float[] _parameters;

        /// <summary>
        ///     Lists to keep track of the evolution of the objective function and the
        ///     norm of the gradient. It is useful to observe also the norm of the gradient 
        ///     because it convergences to 0 when close to a local minimum.
        /// </summary>
        private List<float> _lossVsIteration = new List<float>();
        private List<float> _gradientNormVsIteration = new List<float>();

        /// <summary>
        ///     attributes for generating random initial parameters array
        ///     used in <see cref="InitParameters"/>
        /// </summary>
        Random randomGenerator = new Random();
        private float _lower = -10;
        private float _upper = 10;





        /// <summary>
        ///     Constructor setting a different objective function for each worker,
        ///     The function to minimize by PR-SGD is their average
        /// </summary>
        public Core(LossFunction[] lossFunctions)
        {
            int numWorkers = lossFunctions.Length;

            // initialize the workers
            _workers = new Worker[numWorkers];
            for (int i = 0; i < numWorkers; i++)
                _workers[i] = new Worker(i, lossFunctions[i]);

            // define the average loss function
            _averageLoss = new AverageLoss(lossFunctions);

            // initialize parameters
            var dim = _averageLoss.GetInputDimension();
            _parameters = InitParameters(dim, _lower, _upper);
        }



        /// <summary>
        ///     Constructor setting the same objective function for all the workers
        /// </summary>
        public Core(int numWorkers, LossFunction lossFunction)
        {
            // initialize the workers
            _workers = new Worker[numWorkers];
            for (int i = 0; i < numWorkers; i++)
                _workers[i] = new Worker(i, lossFunction);

            // define the average loss function
            _averageLoss = lossFunction;

            // initialize parameters
            var dim = _averageLoss.GetInputDimension();
            _parameters = InitParameters(dim, _lower, _upper);
        }




        /// <summary>
        ///     Returns an array of size dim, where each coordinate is a value 
        ///     sampled uniformly in the interval (lower, upper)
        /// </summary>
        private float[] InitParameters(int dim, float lower=-1, float upper=1) 
        {
            float[] parameters = new float[dim];
            for (int i = 0; i < dim; i++)
                parameters[i] = Uniform(lower, upper);
            return parameters;
        }




        /// <summary>
        ///     Runs asynchronously the algorithm PR-SGD during numIters iteration, 
        ///     and during each iteration, each worker workers[i] performs numStepsWorker[i]
        ///     steps of LocalSgd
        /// and keeps track of the loss evolution
        /// </summary>
        public async Task RunPRSGD(int numIters, int[] numLocalSteps, float learningRate)
        {
            var workerTask = new Task[_numWorkers]; 
            for (int iter = 0; iter < numIters; iter++)
            {
                // Local Sgd
                //----------
                for (var i = 0; i < _numWorkers; i++)
                {
                    var worker = _workers[i];
                    var numSteps = numLocalSteps[i];
                    worker.SetParameters(_parameters);
                    workerTask[i] = Task.Run(() => worker.LocalSGD(learningRate, numSteps));
                }


                // Wait for all the workers to finish
                //-----------------------------------
                foreach (var task in workerTask)
                    await task;


                // Average the results
                //--------------------
                Array.Clear(_parameters, 0, _parameters.Length);
                foreach (var worker in _workers)
                {
                    var workerParameters = worker.GetParameters();
                    for (var i = 0; i< _parameters.Length; i++)
                    {
                        _parameters[i] += workerParameters[i] / _numWorkers;
                    }
                }

                // Evaluate the loss and the gradient's norm
                //------------------------------------------
                _lossVsIteration.Add(_averageLoss.Value(_parameters));
                _gradientNormVsIteration.Add(Norm(_averageLoss.Gradient(_parameters, deleteNoise:true)));

            }
        }







        /// <summary>
        ///     Shows the loss evolution until the last iteration
        /// </summary>
        public void ShowLoss()
        {
            Console.WriteLine("Loss evolution");
            foreach (var lossVal in _lossVsIteration)
                Console.Write("{0}, ", lossVal);

            Console.WriteLine("\n");

            Console.WriteLine("Gradient Norm evolution");
            foreach (var lossVal in _gradientNormVsIteration)
                Console.Write("{0}, ", lossVal);

            Console.WriteLine();
        }

        /// <summary>
        ///     returns a float chosen uniformly at random in (a,b)
        /// </summary>
        private float Uniform(float a, float b)
        {
            var rand = (float)randomGenerator.NextDouble();
            return a + (b - a) * rand;
        }

        /// <summary>
        ///     compute the 2-norm of an array
        /// </summary>
        public float Norm(float[] arr)
        {
            float norm = 0.0f;
            foreach (var coord in arr)
                norm += coord * coord;
            return (float)Math.Sqrt(norm);
        }

    }
}
