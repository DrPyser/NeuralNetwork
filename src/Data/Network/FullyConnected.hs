{-# LANGUAGE BangPatterns #-}
-- # LANGUAGE UndecidableInstances #
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Data.Network.FullyConnected
  (FeedForwardFC, FullyConnected(FC), Layers,
   weights, biases, activation,
   unGrad, unGradBatch, Grad(GradBatch, Grad))
where

import Numeric.LinearAlgebra as LA hiding (fromList, toList)
import Numeric.LinearAlgebra.Devel
import qualified Data.Vector.Storable as V hiding (foldr, modify)
import System.Random
import Control.DeepSeq
-- import Data.Foldable as F
import Data.Bifunctor
import Data.Network as Network
import Data.Network.Activation
import Algorithm.Training
import Algorithm.GradientDescent 
import Data.Network.Utils (squeeze, (.*))


-- Fully connected feedforward layer
data FullyConnected i a = FC {weights :: {-# UNPACK #-} !(Matrix a),
                              biases ::  {-# UNPACK #-} !(Vector a),
                              activation :: ActivationFunction i
                       }

instance (V.Storable a, NFData a) => NFData (FullyConnected i a) where
  rnf (FC w b a) = rnf w `deepseq` rnf b

instance Layer (Vector R) (FullyConnected (Vector R) R) where
  compute (FC weights biases _) input = biases + (input <# weights)
  run     l@(FC weights biases af) input = unAct af (compute l input)
  makeRandomLayer g rf i o af = let rnds = (randomRs (0,1) g) :: [Double]
                                    rnds' = (rf rnds) 
                                    rw   = (i><o) rnds'
                                    rb   = o |> (repeat 0)
                                in (FC rw rb af)

instance Layer (Matrix R) (FullyConnected (Matrix R) R) where
  compute (FC weights biases _) input = (fromRows $ replicate (rows input) biases) + (input <> weights)
  run     l@(FC _ _ af) input = unAct af (compute l input)
  makeRandomLayer g rf i o af = let rnds = (randomRs (0,1) g) :: [Double]
                                    rnds' = (rf rnds) 
                                    rw   = (i><o) rnds'
                                    rb   = o |> (repeat 0)
                                in (FC rw rb af)


instance (Container Matrix a, Container Vector a, Num a) => Show (FullyConnected i a) where
  show l = let (a,b) = size $ weights l
           in "Layer " ++ (show a) ++ "x" ++ (show b) ++ " " ++ (name $ activation l)

instance Floating a => Num (FullyConnected a Double) where
  (FC w b a) + (FC w' b' _) = FC (w + w') (b + b') a
  (FC w b a) - (FC w' b' _) = FC (w - w') (b - b') a
  (FC w b a) * (FC w' b' _) = FC (w * w') (b * b') a
  abs (FC w b a) = FC (abs w) (abs b) a
  signum (FC w b a) = FC (signum w) (signum b) a
  fromInteger x = FC (fromInteger x) (fromInteger x) (act id "identity")

instance Floating a => Fractional (FullyConnected a Double) where
  (FC w b a) / (FC w' b' _) = FC (w / w') (b / b') a
  recip (FC w b a) = FC (recip w) (recip b) a
  fromRational a = FC (fromRational a) (fromRational a) (act id "identity")

instance (Container Vector a, Container Matrix a) => Linear a (FullyConnected i) where
  scale x (FC w b a) = FC (scale x w) (scale x b) a

-- Container Layer instance?

instance (Layer i (FullyConnected i Double)) => Network i (Layers (FullyConnected i Double)) where
  runNetwork (Output l) i = run l i
  runNetwork (l :=> n)  i = runNetwork n (fst $ run l i)
  runLayers  (Output l) i = [run l i]
  runLayers  (l :=> n)  i = let y = (run l i)
                            in  y:(runLayers n (fst y))

  makeRandomNetwork g t nt i [] (o, af) = let l = makeRandomLayer g t i o af
                                          in  (Output l)
  makeRandomNetwork g t nt i ((h,haf):hs) o = let (g1,g2) = split g
                                                  l = makeRandomLayer g1 t i h haf
                                              in nt $ l :=> (makeRandomNetwork g2 t nt h hs o)

type FeedForwardFC i = Layers (FullyConnected i Double)

instance (Monad m) => Backpropagation Data (NetworkEvaluator Data) (FeedForwardFC Data) m where
  backward lf n (out,tar) das = do
    let δout = (derivate lf (tar, out)) -- dE/dy
        deltas = scanr (\(l, a') δ -> let w = weights l in a' * (w #> δ)) δout (zip (tail $ toList n) das)
    return (deltas) -- deltas for computing bias

instance (Monad m) => Backpropagation Batch (NetworkEvaluator Batch) (FeedForwardFC Batch) m where
  backward lf n (out,tar) das = do
    let δout = tr (derivate lf (tar, out)) -- dE/dy
        deltas = scanr (\(l, a') δ -> let w = weights l in (tr a') * (w <> δ)) δout (zip (tail $ toList n) das)
    return (deltas) -- deltas for computing bias

-- type instance Params lf = forall i. ((FeedForwardFC i), (i,i))

instance (Monad m) => GradientDescent (NetworkEvaluator Batch) ((FeedForwardFC Batch), (Batch,Batch)) m where
  -- parameters of loss function: Network(weights, biases, activation functions for each layer)
  -- and sample((input, output) pair)
    
  data Grad (NetworkEvaluator Batch) = GradBatch { unGradBatch :: ([Matrix Double], [Vector Double]) } deriving (Show)

  -- Gradients over parameters(weights, biases)
  grad lf (n, (i,t)) = do
    let aos = runLayers n i -- forward propagation: compute layers outputs and activation derivatives        
        (as, as') = (unzip) aos 
        (out) = last as
    (ds) <- backward lf n (out, t) (init as') -- compute gradients with backpropagation
    let gs = zipWith (\δ a -> tr (δ <> a)) ds (i:init as)
    return $ GradBatch (gs, squeeze <$> ds)

  move lr (n, (i,t)) (GradBatch (gs, ds)) = do
    -- update function
    -- divide by batch size
    let r = fromIntegral $ rows i
        update = (\(FC w b af) g δ -> FC (w + (lr/r).*g) (b + (lr/r).*δ) af)
        n' = Network.fromList $ zipWith3 update (Network.toList n) gs ds -- new model
    return (n', (i,t))

instance (Monad m) => GradientDescent (NetworkEvaluator Data) ((FeedForwardFC Data), (Data,Data)) m where
  -- parameters of loss function: Network(weights, biases, activation functions for each layer)
  -- and sample((input, output) pair)
    
  data Grad (NetworkEvaluator Data) = Grad { unGrad :: ([Matrix Double], [Vector Double]) } deriving (Show)

  -- Gradients over parameters(weights, biases)
  grad lf (n, (i,t)) = do
    let aos = runLayers n i -- forward propagation: compute layers outputs and activation derivatives        
        (as, as') = (unzip) aos -- juggle data to extract layers inputs and derivatives in right order
        out = last as
    (ds) <- backward lf n (out,t) (init as') -- compute gradients with backpropagation
    let gs = zipWith (\δ i -> tr (δ `outer` i)) ds (i:as)
    return $ Grad (gs, ds)

  move lr (n, (i,t)) (Grad (gs, ds)) = do
    -- update function
    let update = (\(FC w b af) g δ -> FC (w + lr.*g) (b + lr.* δ) af)
        n' = Network.fromList $ zipWith3 update (Network.toList n) gs ds -- new model
    return (n', (i,t))

