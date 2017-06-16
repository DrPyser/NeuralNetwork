{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ViewPatterns #-}
module Algorithm.Training where

import Prelude hiding (foldl, foldr)
-- import Numeric.AD
import Numeric.LinearAlgebra as LA hiding (fromList, (<>))
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector.Storable as V hiding (foldr, modify)
import Data.Functor
import Data.Bifunctor
import Data.Monoid as M
import Data.Foldable as F
import Data.Ord
import Control.Monad
import Pipes
import Pipes.Prelude as P
import Control.DeepSeq
-- import Control.Monad.RWS
-- import Control.Monad.RWS as RWS 
import Data.Network.Activation
import Data.Network
import Data.Network.Utils
import Algorithm.GradientDescent


-- type TrainingSet = [Sample]

-- type TestSet = [Sample]

-- type ResultSet = [(Output, Output)]

-- Loss function parameterized by sample type (a), output loss type(l) and parameter space (p)
data LossFunction a l = Evaluator { evaluate :: (a, a) -> l, evaluate' :: (a, a) -> l }

instance Differentiable (a,a) l (LossFunction a l) where
  derivate = evaluate' 

-- The type of a loss/error evaluation
type Loss = Vector R
type LossBatch = Matrix R

-- y is target, y' is actual
mse :: (Floating a) => LossFunction a a
mse = let f (y,y') = let γ = y'-y in (/ 2) $ γ**2
          f' (y,y') = (y'-y)
      in  Evaluator f f'

mseBatch :: LossFunction Batch Batch
mseBatch = let fb (b, b')  = (/ 2) $ (**2) (b' - b)
               fb' (b, b') = (b' - b)
           in Evaluator fb fb' 

type NetworkEvaluator a = LossFunction a a

type LearningRate = Double
-- Training parameters = (epochs, learning rate)
type TrainingParameters = (Int, Double)

class (Network i n, Differentiable (i, i) i lf, Monad m) => Backpropagation i lf n m | lf -> i, n -> i where
  -- given a neural network and an input, compute each layer's activation and derivative
  -- forward :: n -> i -> m [(i,i)] 
  -- given a loss function, a network, an output and target pair and a list of activation derivatives, compute the deltas
  backward :: lf -> n -> (i, i) -> [i] -> m [i]


data TrainingState m  = TS {
  tsmodel :: m,      -- the model(network) state
  tseta   :: Double, -- the learning rate η
  tsiteration :: Int,-- the iteration number(one per gradient move)
  tsloss  :: Double,  -- the loss of the model over the last training batch/sample
  tssampleid :: Int
  } deriving Show

-- type DescentStopper i n = StopCondition (LossFunction i i) (n, (i,i))

class AsDouble a where
  toDouble :: a -> Double

instance AsDouble (Vector R) where
  toDouble = norm_2

instance AsDouble (Matrix R) where
  toDouble = norm_2

-- A pipe that takes a sample and makes one step in the parameter space
stepP :: (Monad m, AsDouble i, Network i n, GradientDescent (LossFunction i i) (n, (i,i)) m) =>
  n -> LossFunction i i -> Int -> (Int -> Double) -> (Int, (i,i)) -> Producer (TrainingState n) m r
stepP m lf n lrf (sid, (i, t)) = loop m n where
  loop m n = do
--    (i, t) <- await -- get a sample
    let !η = lrf n -- compute learning rate for iteration
    (m', _) <- lift $ descend lf η (m, (i,t)) -- step    
    let o = fst $ runNetwork m' i -- test 
        loss = evaluate lf (t,o) -- evaluate
    m' `deepseq` yield $ TS m' η n (toDouble loss) sid -- yield new state
    loop m' (n+1)

-- A pipe that produce a reporting action on a training state
reportP :: (Monad m, AsDouble i, Network i n) => Int -> (TrainingState n -> m ()) -> Pipe (TrainingState n) (TrainingState n) m r
reportP interval action = P.mapM $ \ts -> do
  when (tsiteration ts `mod` interval == 0) (action ts)
  return ts

-- Does gradient descent given a stop condition
descendP :: (Monad m, AsDouble i, Network i n) =>
   (TrainingState n -> TrainingState n -> m (Maybe x)) -> Consumer (TrainingState n) m x
descendP stopper = loop
  where loop = do
          ts <- await
          ts' <- await
          mx <- lift (stopper ts ts')
          case mx of
            Just x -> return x
            Nothing -> loop


-- A stop condition: stop if the loss doesn't change significantly enough,
-- returning the state with the minimum loss
closeEnough :: (Network i n, Monad m) => Double -> (TrainingState n -> TrainingState n -> m (Maybe (TrainingState n)))
closeEnough tolerance ts ts' = do
  if abs (tsloss ts' - tsloss ts) < tolerance then
    return (Just $ minimumBy (comparing tsloss) [ts, ts']) else return Nothing
    
longEnough :: (Network i n, Monad m) => Int -> (TrainingState n -> TrainingState n -> m (Maybe (TrainingState n)))
longEnough iteration ts ts' = if tsiteration ts' >= iteration then
  return (Just $ minimumBy (comparing tsloss) [ts, ts']) else return Nothing


improvement :: (Network i n, Monad m) => Double -> (TrainingState n -> TrainingState n -> m (Maybe (TrainingState n)))
improvement tolerance ts ts' = if ((tsloss ts' - tsloss ts) > tolerance) then
  return (Just $ minimumBy (comparing tsloss) [ts, ts']) else return Nothing 





