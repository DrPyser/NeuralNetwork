{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Data.Network.Activation (ActivationFunction, unAct, act, activate, derivate, logistic, softmax, softmaxMat, relu) where

import Numeric.AD.Mode.Kahn
import Numeric.LinearAlgebra
import Data.Network.Utils(Differentiable, squeeze, derivate, (.*))

data ActivationFunction a = Act {unAct :: (a -> (a,a))}

act :: Floating a => (forall s. AD s (Kahn a) -> (AD s (Kahn a))) -> ActivationFunction a
act f = Act (diff' f)

activate :: ActivationFunction a -> a -> a
activate af = fst . (unAct af)

instance Differentiable a a (ActivationFunction a) where
  derivate af = snd . (unAct af)

-- derivate :: ActivationFunction a -> a -> a
-- derivate af = snd . (unAct af)



logistic :: Floating a => ActivationFunction a
logistic = let f x =  1 / (1 + exp (-x))
           in  act f


deltaf :: Floating a => Int -> Int -> a
deltaf i j = if (i == j) then 1 else 0

softmax :: (Fractional e, Linear e Vector, Floating (Vector e), Container Vector e) => ActivationFunction (Vector e)
softmax = let f x = let n = sumElements (exp x)
                    in (recip n) .* (exp x)
              f' x = 1 -- Softmax is used only for last layer, so derivative is not needed.
          in  Act (\x -> (f x, f' x))

-- Matrix version for batch computing
softmaxMat :: (Floating e, Container Matrix e, Num (Matrix e), Numeric e, Floating (Matrix e), Floating (Vector e)) =>
  ActivationFunction (Matrix e)
softmaxMat = let f x = let n = fromColumns (replicate (cols x) $ squeeze (exp x))
                       in (recip n) * (exp x)
                 f' x = 1 -- Softmax is used only for last layer, so derivative is not needed.
          in  Act (\x -> (f x, f' x))

relu = let f x = max x 0
       in  act f

-- We can add activation functions and their derivative using the "diff" function from Numeric.AD
