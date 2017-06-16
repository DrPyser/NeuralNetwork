{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Codec.Compression.GZip (decompress)
import Numeric.LinearAlgebra as V
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString as B
import Data.Bifunctor
import Data.Functor
import Data.Word8
import Control.Monad
-- import Control.Monad.State
-- import Control.Monad.Writer
import Pipes as P
import qualified Pipes.Prelude as PP
import System.Random
import Parser
import Data.Network
import Data.Network.FullyConnected
import Algorithm.Training
import Data.Network.Activation
import Data.Network.Utils (makeBatchP, repeatM, squeeze, unzipP, integersP)

-- import System.Random
-- import Data.Binary.Get
-- import Control.Monad.Except as E
-- import qualified Data.Vector as V

main = do
  putStrLn "Extracting MNIST dataset"
  s <- decompress <$> BS.readFile "train-images-idx3-ubyte.gz" -- images data lazy bytestring
  l <- decompress <$> BS.readFile "train-labels-idx1-ubyte.gz" -- labels data lazy bytestring

  s' <- decompress <$> BS.readFile "t10k-images-idx3-ubyte.gz" -- images data lazy bytestring
  l' <- decompress <$> BS.readFile "t10k-labels-idx1-ubyte.gz" -- labels data lazy bytestring

  let mnisttraining = parseMNISTSet s l
  case mnisttraining of
    Right ((nrows, ncols), nsamples, samples) -> do
      putStrLn "MNIST training dataset succesfully parsed!"
      putStrLn $ show nsamples ++ " training samples of " ++ show nrows ++ "x" ++ show ncols ++ " labeled images."
      -- let s = take 100 samples
      -- forM_ s $ \(im, l) -> do      
      --   putStrLn $ renderImage (nrows, ncols) im
      --   putStrLn $ "Target: " ++ show l
      --   putStrLn $ "Label: " ++ show (V.maxIndex l)
      let mnisttest = parseMNISTSet s' l'
      case mnisttest of
        Right ((nrows', ncols'), nsamples', samples') -> do
          putStrLn "MNIST test dataset succesfully parsed!"
          putStrLn $ show nsamples' ++ " test samples of " ++ show nrows' ++ "x" ++ show ncols' ++ " labeled images."

          let batchsize = 1
              -- Sample producer
              trainingSamplesP = each samples :: Producer MNISTSample IO ()
              (imageP, labelP) = unzipP trainingSamplesP :: (Producer Image IO (), Producer Label IO ())
              -- Batch sample producer(pairs of matrix)
              trainingBatches = PP.zip (imageP >-> (makeBatchP batchsize)) (labelP  >-> (makeBatchP batchsize)) 
          -- runEffect $ for (trainingBatches >-> (PP.take 10)) $ \(a,b) -> do
          --   lift $ print b
          -- let epochs   = 3
          g <- getStdGen
          let !(model :: FeedForwardFC Batch) = (makeRandomNetwork g (id) 784 [(30, logistic)] (10, softmaxMat))
              lf = mseBatch
              lrf = \n -> exp (-(1.0 + fromIntegral n))
              tolerance = 0.0000001
              reporter = print 
              interval = 1
              stopper = closeEnough tolerance
              loop m = do
                s@(sid, (i,t)) <- await
                lift $ putStrLn $ "Input " ++ show (fst s)
                lift $ putStrLn $ "Prediction: " ++ show (fst $ runNetwork m i)
                r <- lift $ runEffect $ (stepP m lf 0 lrf s) >-> reportP interval reporter >-> (descendP (stopper))
                lift $ putStrLn $ "Loss vector: " ++ show (evaluate lf (t, fst $ runNetwork (tsmodel r) i))
                yield r
                loop (tsmodel r)
              trainingChain = PP.zip (integersP 0) trainingBatches >-> (PP.take 100) >-> (loop model)

          putStrLn "Model: "
          -- mapM_ ((>> putStrLn "") . print) (map weights $ Network.toList model)
          runEffect $ for trainingChain $ \tsminibatch ->
            lift $ print tsminibatch
          -- runEffect $ for trainingChain $ \ts -> do
          --   lift $ print ts
          -- let trainingRoom = replicateM 3 (last <$> forM samplesb' trainOnceBatch)
          -- putStrLn "Beginning training"
          -- let logresults   = runStateT trainingRoom ((3, 0.5), mseBatch, model)
          -- ((l :: [LossBatch], (_,_,m)), log) <- runWriterT logresults
          -- forM_ log $ \(str, l) -> do
          --   putStrLn $ str ++ ": " ++ show l        
          -- putStrLn $ "Final loss: " ++ (show $ norm_2 $ last l)
      
      putStrLn "Done!"
    Left er -> putStrLn er
  
    
  
