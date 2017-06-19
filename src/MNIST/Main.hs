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
import Data.Network.Utils (makeBatchP, repeatM, squeeze, unzipP, integersP, toGaussian)

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
          g <- getStdGen
          -- let xavier = fmap (\(FC w b af) -> FC (scale (sqrt $ recip $ fromIntegral $ (uncurry (+)) (size w)) w) b af)
          let !(model :: FeedForwardFC Batch) = (makeRandomNetwork g id id 784 [(30, logistic)] (10, softmaxMat))
              -- !model' = fmap (\(FC w b af) -> let (u,_,_) = svd w in (FC u b af)) model
              lf = mse
              lrf = const 1 -- \n -> exp (-(1.0 + fromIntegral n))
              tolerance = 0.0001
              reporter (i,t) ts = do
                putStr "."
                -- putStrLn "Weights : "
                -- mapM_ (print . fst) $ runLayers (tsmodel ts) i
                -- putStrLn "Gradients: "
                -- GradBatch (gs, ds) <- grad lf (tsmodel ts, (i,t))
                -- mapM_ (putStrLn . disps 5) gs
                -- mapM_ print ds
              -- predict = \i ts -> mapM_ (print . fst) $ runLayers (tsmodel ts) i
              interval = 10
              stopper = (uncurry (liftM2 (<|>)) . ((closeEnough tolerance) &&& (longEnough 100)))
              loop m = do -- the actual training steps
                s@(sid, (i,t)) <- await
                lift $ putStrLn $ "Input " ++ show (fst s)
                -- lift $ putStrLn $ "Biases " ++ show (map biases $ Data.Network.toList m)
                -- lift $ putStrLn $ "Weights " ++ show (map weights $ Data.Network.toList m)
                -- lift $ putStrLn $ "Initial Prediction: " ++ show (map fst $ runLayers m i)
                -- g <- PP.map (grad lf) (m, (i,t))
                -- (m')
                r <- lift $ runEffect $ (stepP m lf 0 lrf s) >-> reportP interval (reporter (i,t)) >->
                  (descendP (stopper))
                -- let r = runNetwork m i
                -- let l = toDouble $ evaluate lf (t, fst r)
                yield r
                -- lift $ mapM_ ((>> putStrLn "") . print) (map weights $ Data.Network.toList (tsmodel r))
                loop (tsmodel r)
              
              trainingChain m = PP.zip (integersP 0) trainingBatches >-> (PP.take 10000) >-> (loop m)
              loop' m n = do
                (Just r) <- lift $ PP.last (trainingChain m)
                yield (n, r)
                loop' (tsmodel r) (succ n)
                
          runEffect $ for ((loop' model 0) >-> (PP.take 10)) $ \(n,ts) -> do
            lift $ putStrLn $ "Epoch: " ++ show n 
            lift $ print ts

          -- runEffect $ for (trainingBatches >-> (PP.take 10)) $ \(i,l) -> do
          --   lift $ putStrLn $ disps 5 i
          --   lift $ print l
          
      putStrLn "Done!"
    Left er -> putStrLn er
  
    
  
