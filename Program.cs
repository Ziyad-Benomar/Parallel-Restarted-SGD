namespace PRSGD
{
    public class Program
    {
        public static async Task Main()
        {
            int dim = 256;
            int numWorkers = 4;


            // The loss function
            var centers = new float[dim];
            var lossFunctions = new LossFunction[numWorkers];
            for (int w = 0; w < numWorkers; w++)
            {
                // Create a loss function for the worker w
                for (int i = 0; i < dim; i++)
                    centers[i] = i + w*w;
                // centers is cloned inside the constructor of QuadraticLoss
                // ==> we can modify the same centers array here to create a new QuadraticLoss
                lossFunctions[w] = new QuadraticLoss(dim, centers, noisyGrads:true);
            }
            var avgLoss = new AverageLoss(lossFunctions);


            // PR-SGD hyperparameters
            int numIterations = 50;
            var numLocalSteps = new int[numWorkers];
            for (int i = 0; i < numWorkers; i++)
                numLocalSteps[i] = 100;
            float learningRate = 0.01f;


            // Run PR-SGD
            var core = new Core(numWorkers, avgLoss);
            await core.RunPRSGD(numIterations, numLocalSteps, learningRate);
            core.ShowLoss();
        }


    }
    

}
