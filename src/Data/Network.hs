{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}

module Data.Network
  (Network, Layer, Batch, Data, Layers(Output, (:=>)),
   makeRandomLayer, makeRandomNetwork, run, compute, runLayers, runNetwork, fromList, toList
  )
  --   makeRandomNetwork, makeRandomModel, makeModel, fromList,
  --   runLayer, runLayer', runNetwork, runLayers, runLayers',
  --   runLayerBatch, runLayerBatch', runNetworkBatch, runLayersBatch, runLayersBatch',
  --   Input, Output, Data, Batch, BatchIn, BatchOut, Sample, BatchSample, Weights)
where

import GHC.Exts hiding ((<#))
import Prelude hiding (scanl)
import Data.Functor
-- import Data.Word8
import Data.Foldable hiding (toList)
import Data.Monoid
import System.Random
import Control.Monad.Random as R hiding (fromList)
import Control.Monad
import Control.DeepSeq
import Numeric.LinearAlgebra as LA hiding ((<>), fromList, toList)
import Numeric.LinearAlgebra.Data as LAD hiding (fromList, toList, size, R)
-- import Numeric.LinearAlgebra.Static as LAS hiding ((<>), fromList)
import Numeric.LinearAlgebra.Devel
import Data.Network.Utils (scanl)
import Data.Network.Activation

-- type ActivationFunction = forall a. Floating a => a -> a

type Input = Vector Double
type Output = Vector Double
type Data = Vector Double
type Weights = Matrix Double
type Batch = Matrix Double
type BatchIn = Matrix Double
type BatchOut = Matrix Double
type Sample = (Data, Data) -- Labeled data for supervised learning
type BatchSample = (Batch, Batch)
-- type Model a = (Network, ActivationFunction a) -- A network's structure with its activation function

-- Network layer = (weight matrix, bias vector)
-- Parameter a: type of data(e.g. Double)

-- What's a layer?
class (NFData a) => Layer i a | a -> i where  
  compute :: a -> i -> i
  run     :: a -> i -> (i, i)
  
  makeRandomLayer :: (RandomGen g) => g -> ([Double] -> [Double]) -> Int -> Int -> ActivationFunction i -> a

-- What's a network?
class (NFData a) => Network i a | a -> i where
  runNetwork :: a -> i -> (i,i)
  runLayers  :: a -> i -> [(i,i)]
  makeRandomNetwork :: (RandomGen g) =>
    g -> ([Double] -> [Double]) -> Int -> [(Int, ActivationFunction i)] -> (Int, ActivationFunction i) -> a

-- Linked list Structure for feedforward neural network
data Layers :: * -> * where
  Output :: !a -> Layers a
  (:=>) :: !a -> !(Layers a) -> Layers a 
  
infixr 5 :=>

instance (NFData a) => NFData (Layers a) where
  rnf (Output x) = rnf x
  rnf (l :=> ll) = rnf l `seq` rnf ll
  
instance Show a => Show (Layers a) where
  show (Output l) = "(Output) " ++ show l
  show (a :=> b) = show a ++ " -> " ++ show b

instance Functor Layers where
  fmap f (Output x) = (Output $ f x)
  fmap f (a :=> b) = (f a) :=> (fmap f b)
  
instance Foldable Layers where
  foldr f b (Output x) = f x b
  foldr f c (a :=> b) = f a (foldr f c b)

  foldl f a (Output x) = f a x
  foldl f a (b :=> c) = foldl f (f a b) c

instance Num a => Num (Layers a) where
  (Output l) + (Output r) = Output (l + r)
  (l :=> n) + (r :=> n') = (l + r) :=> (n + n')
  (Output l) - (Output r) = Output (l - r)
  (l :=> n) - (r :=> n') = (l - r) :=> (n - n')
  (Output l) * (Output r) = Output (l * r)
  (l :=> n) * (r :=> n') = (l * r) :=> (n * n')
  abs (Output l) = Output (abs l)
  abs (l :=> n) = (abs l :=> abs n)
  signum (Output l) = Output (signum l)
  signum (l :=> n) = (signum l) :=> (signum n)
  fromInteger a = Output (fromInteger a)

instance Fractional a => Fractional (Layers a) where
  (Output l) / (Output r) = Output (l/r)
  (l :=> n) / (r :=> n') = (l/r) :=> (n/n')
  recip (Output l) = Output (recip l)
  recip (l :=> n) = (recip l) :=> (recip n)
  fromRational a = Output (fromRational a)

    -- g -> ([Double] -> [Double]) -> Int -> [(Int, ActivationFunction i)] -> (Int, ActivationFunction i) -> a


instance IsList (Layers i) where
  type Item (Layers i) = i
  fromList []     = undefined
  fromList [x]    = Output x
  fromList (x:xs) = x :=> (fromList xs)

  toList (Output x) = [x]
  toList (x :=> y)  = x:(toList y)

-- fromList :: [a] -> Layers a
-- fromList [] = undefined
-- fromList [x] = Output x
-- fromList (x:xs) = x :=> (fromList xs)

-- make a fully connected layer; m = size of input(previous layer), n = size of output(next layer)
-- makeLayerFC :: Floating a => Int -> Int -> ActivationFunction a -> Layer a Double
-- makeLayerFC m n af = Layer { weights = (m><n) (repeat 0), biases = n |> (repeat 0), activation = af }

-- make a fully connected layer with randomly generated (normally distributed) weights and biases
-- makeLayerFCR :: MonadRandom m => RandDist -> Int -> Int -> ActivationFunction a -> m (Layer a Double)
-- makeLayerFCR dist m n = do
--   (seeds :: [Int]) <- getRandoms
--   (seed2 :: Int) <- getRandom
--   let w = fromRows $ fmap (\s -> randomVector s dist n) (take m seeds)
--       b = randomVector seed2 dist n
--   return $ Layer { weights = w, biases = b }
  
-- makeRandomNetwork :: MonadRandom m => RandDist -> Int -> [Int] -> Int -> m Network
-- makeRandomNetwork dist i [] o = Output <$> (makeLayerFCR dist i o)
-- makeRandomNetwork dist i (h:hs) o = (:=>) <$> makeLayerFCR dist i h <*> makeRandomNetwork dist h hs o

-- makeModel :: Network -> [ActivationFunction a] -> Model a
-- makeModel (Output l) (a:afs) = Output (l,a)
-- makeModel (l :=> n) (a:afs) = ((l,a) :=> makeModel n afs)

-- makeRandomModel :: MonadRandom m => RandDist -> Int -> [(Int, ActivationFunction a)] -> (Int, ActivationFunction a) -> m (Network i)
-- makeRandomModel dist i [] (o, oaf) = (Output . (,oaf)) <$> (makeLayerFCR dist i o)
-- makeRandomModel dist i ((h,af):hs) (o, oaf) = ((:=>) . (,af)) <$> makeLayerFCR dist i h <*> makeRandomModel dist h hs (o, oaf)

-- compute layer's net result (before activation/squashing)
-- computeLayer :: Layer a Double -> a -> Vector Double
-- computeLayer (Layer weights biases) input = biases + input <# weights 

-- net linear combiner for a batch of input as a matrix, with rows as input vectors
-- computeLayerBatch :: Layer Batch Double -> Batch -> Batch
-- computeLayerBatch (Layer weights biases) inputs = (fromRows $ replicate (rows inputs) biases) + (inputs <> weights)

-- run layer with input vector
-- runLayer :: ActivationFunction Output -> Layer Double -> Input -> Output
-- runLayer af l i = activate af $ computeLayer l i

-- run layer with output vector, outputing both result and derivative
-- runLayer' :: ActivationFunction Output -> Layer Double -> Input -> (Output, Output)
-- runLayer' af l i = unAct af $ computeLayer l i

-- Batch variants

-- runLayerBatch :: ActivationFunction Batch -> Layer Double -> BatchIn -> BatchOut
-- runLayerBatch af l i = activate af $ computeLayerBatch l i

-- runLayerBatch' :: ActivationFunction Batch -> Layer Double -> BatchIn -> (BatchOut, BatchOut)
-- runLayerBatch' af l i = unAct af $ computeLayerBatch l i

-- run network on input vector with an activation function
-- runNetwork :: Model Data -> Input -> Output
-- runNetwork n i = foldl' ((flip . uncurry . flip) runLayer) i n

-- runNetworkBatch :: Model Batch -> BatchIn -> BatchOut
-- runNetworkBatch n i = foldl' ((flip . uncurry . flip) runLayerBatch) i n

-- run network on input, accumulating each layer's intermediary result,
-- with the first element of the resulting list being the last layer's output, ...
-- runLayers :: Model Data -> Input -> [Output]
-- runLayers n i = foldl' (\os@(o:os') (l,af) -> (runLayer af l o):os) [i] n

-- Same as above but with derivatives
-- runLayers' :: Model Data -> Input -> [(Output, Output)]
-- runLayers' n i = foldl' (\os@((o, o'):os') (l, af) -> (runLayer' af l o):os) [(i, mempty)] n

-- Batch variants
-- runLayersBatch :: Model Batch -> BatchIn -> [BatchOut]
-- runLayersBatch n i = foldl' (\os@(o:os') (l, af) -> (runLayerBatch af l o):os) [i] n

-- runLayersBatch' :: Model Batch -> BatchIn -> [(BatchOut, BatchOut)]
-- runLayersBatch' n i = foldl' (\os@((o, o'):os') (l, af) -> (runLayerBatch' af l o):os) [(i, mempty)] n

