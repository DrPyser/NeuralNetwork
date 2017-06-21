{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Codec.Compression.GZip (decompress)
import Numeric.LinearAlgebra as V
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString as B
import Data.Bifunctor
import Data.Functor
import Data.Word8
import qualified Data.Vector.Storable as V
import Control.Monad
import Control.Arrow
import Control.Applicative
import Pipes as P
import qualified Pipes.Prelude as PP
import System.Random
import Parser
import Data.Network
import Data.Network.FullyConnected as FC
import Algorithm.Training
import Algorithm.GradientDescent
import Data.Network.Activation
import Data.Network.Utils (makeBatchP, repeatM, squeeze, unzipP, integersP, toGaussian, averageP, accuracyBatch)

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
          
          let batchsize = 20
              -- Sample producer
              trainingSamplesP = each samples :: Producer MNISTSample IO ()
              (imageP, labelP) = unzipP trainingSamplesP :: (Producer Image IO (), Producer Label IO ())
              -- Batch sample producer(pairs of matrix)
              trainingBatches = PP.zip (imageP >-> (makeBatchP batchsize)) (labelP  >-> (makeBatchP batchsize))
              testSamplesP    = each samples' :: Producer MNISTSample IO ()
              testBatches     = uncurry PP.zip $ bimap (>-> makeBatchP batchsize) (>-> makeBatchP batchsize) (unzipP testSamplesP)
          g <- getStdGen
          -- let xavier = fmap (\(FC w b af) -> FC (scale (sqrt $ recip $ fromIntegral $ (uncurry (+)) (size w)) w) b af)
          let !(model :: FeedForwardFC Batch) = (makeRandomNetwork g id id 784 [(1000, logistic)] (10, softmaxMat))
              -- !model' = fmap (\(FC w b af) -> let (u,_,_) = svd w in (FC u b af)) model
              lf = mse
              lrf = const 0.05 -- \n -> exp (-(1.0 + fromIntegral n))
              tolerance = 0.0001
              test m = do
                (i,t) <- await
                let r = runNetwork m i
                    loss = evaluate lf (t, fst r)
                    a    = accuracyBatch (fst r, t)
                yield ((toDouble loss), a)
              reporter (i,t) ts = do
                putStrLn $ show ts
              interval = 10
              stopper = (uncurry (liftM2 (<|>)) . ((closeEnough tolerance) &&& (longEnough 100)))
              loop m = do -- the actual training steps
                s@(sid, (i,t)) <- await
                -- lift $ putStrLn $ "Input " ++ show (fst s)
                r <- lift $ runEffect $ (stepP m lf 0 lrf s) >-> reportP interval (reporter (i,t)) >->
                  (descendP (stopper))
                -- lift $ putStrLn ""
                yield r
                loop (tsmodel r)
              
              trainingChain m = PP.zip (integersP 0) trainingBatches >-> (PP.take 100) >-> (loop m)
              -- Train over all training samples, then test with testing dataset, for each iteration
              -- One iteration = one epoch
              loop' m n = do
                lift $ putStrLn $ "Epoch: " ++ show n 
                (Just r) <- lift $ PP.last (trainingChain m)
                yield r
                let (lossP, accuracyP) = unzipP $ testBatches >-> test (tsmodel r)
                (loss, accuracy) <- lift $ liftM2 (,) (averageP lossP) (averageP accuracyP)
                lift $ putStrLn $ "Test loss: " ++ show (fst loss)
                lift $ putStrLn $ "Test accuracy: " ++ show (fst accuracy)
                loop' (tsmodel r) (succ n)
                
          runEffect $ for ((loop' model 0) >-> (PP.take 100)) $ \ts -> do
            lift $ putStrLn ""
            lift $ print ts
          
      putStrLn "Done!"
    Left er -> putStrLn er
  
    
  
