{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}

module Algorithm.GradientDescent
 (GradientDescent, Grad, StopCondition(StopWhen),
  grad, move, descend
 )
where

-- import Numeric.AD as AD

import Control.Monad
import Pipes

class Monad m => GradientDescent a p m where  
  -- type instance Params a = p
  data Grad a :: (*)
  -- function -> parameters -> gradient of function over parameters
  grad :: a -> p -> m (Grad a)
  -- step size -> initial parameters -> gradients -> new parameters
  move :: Double -> p -> Grad a -> m p 

newtype StopCondition a p = StopWhen (p -> p -> Bool)
                        
instance (Monad m, Floating a) => GradientDescent (a -> a) a m where
  data Grad (a -> a) = Grad { unGrad :: a }

  grad f (p) = (return . Grad) $ (f p - f (p - epsilon)) / epsilon
    where epsilon = (0.1)**10

  -- move scale (Arg p) (Grad g) = (return . Arg) $ p + (fromRational (toRational scale) * g)
  move scale (p) (Grad g) = (return) $ p + (fromRational (toRational scale) * g)

type StepSize = Double

descend :: GradientDescent a p m => a -> StepSize -> p -> m p
descend f eta p = do
  gradients <- grad f p
  move (-eta) p gradients

-- descentP :: GradientDescent a p m => a -> StepSize -> p -> Pipe p p m r
-- gradientDescent f (StopWhen stop) alpha pinit = do
--   p <- step pinit
--   if (stop pinit p) then return p
--     else gradientDescent f (StopWhen stop) alpha p
--   where step params = do
--           gradients <- grad f params
--           move (-alpha) params gradients

  
-- closeEnough :: (Ord a, Floating a) => (a -> a) -> a -> StopCondition (a -> a)
-- closeEnough f tolerance = StopWhen stop
--   where stop (Arg p) (Arg p') = abs (f p - f p') < tolerance



