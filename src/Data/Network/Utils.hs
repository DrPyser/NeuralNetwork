{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Network.Utils
 (repeatM, scanl, randomInput, rowNorm, rowNorm', (.*), (*.),
  Differentiable, derivate, squeeze, squish,
  makeBatchP, unzipP, integersP,
  accuracy)
where

import Prelude hiding (scanl, zip)
import Data.Bifunctor as Bf
import Data.Foldable as F
import Numeric.LinearAlgebra as LA
import Control.Monad.Random as R
import qualified Data.Vector.Storable as V
import Control.Applicative as A
import Pipes as P
import qualified Pipes.Prelude as PP

scanl :: Foldable f => (b -> a -> b) -> b -> f a -> [b]
scanl f b a = reverse $ foldl (\(i:is) x -> (f i x):i:is) [b] a

scanr :: Foldable f => (a -> b -> b) -> b -> f a -> [b]
scanr f b a = foldr (\x (i:is) -> (f x i):i:is) [b] a

randomInput :: MonadRandom m => RandDist -> Int -> m (LA.Vector Double)
randomInput dist i = do
  seed :: Int <- getRandom
  return $ LA.randomVector seed dist i

a .* b = LA.scale a b
a *. b = LA.scale b a

rowNorm :: Matrix Double -> Vector R
rowNorm m = squeeze $ sqrt (m**2) 

rowNorm' :: (Element a, Normed (Vector a)) => Matrix a -> Vector R
rowNorm' = LA.fromList . (fmap norm_2) . toRows

class Differentiable a b c | c -> a b where
  derivate :: c -> (a -> b)

repeatM :: Monad m => Int -> (a -> m a) -> a -> m a
repeatM n f i 
  | n <= 0 = return i
  | otherwise = do
      o <- f i
      repeatM (n-1) f o

-- flatten laterally(summing over rows)
squeeze :: (V.Storable a, Num a, Numeric a) => Matrix a -> Vector a
squeeze m = m #> (V.generate (cols m) (const 1))

-- flatten vertically(summing over columns)
squish :: (V.Storable a, Num a, Numeric a) => Matrix a -> Vector a
squish m = (V.generate (rows m) (const 1)) <# m 

--- Pipe utils
-----------------------------------------------------------------------------------------

-- Collect chunks of values of a particular size and yields those chunks as lists
-- Note: the elements of each list are in reverse order of arrival.
-- Compose with (PP.map reverse) to change that.
chunkP :: (Monad m) => Int -> Pipe a [a] m r
chunkP n = collect n [] where
  collect 0 xs = (yield xs) >> (collect n [])
  collect n xs = do
    x <- await
    collect (n-1) (x:xs)

-- transforms a stream of vectors into a stream of matrix(i.e. batch of vectors)
makeBatchP :: (Element a, Monad m) => Int -> Pipe (Vector a) (Matrix a) m r
makeBatchP n = chunkP n >-> PP.map (LA.fromRows)
  
  
unzipP :: (Monad m) => Producer (a,b) m r -> (Producer a m r, Producer b m r)
unzipP p = (p >-> (PP.map fst), p >-> (PP.map snd))

integersP :: (Monad m, Enum a) => a -> Producer a m r
integersP n = yield n >> integersP (succ n)

-----------------------------------------------------------------------------------

accuracy :: [(Vector R, Vector R)] -> Double
accuracy outs = (/ (fromIntegral $ length outs)) $ fromIntegral $ length $ filter (\(y,y') -> (maxIndex y == maxIndex y')) outs 

accuracyBatch :: [(Matrix R, Matrix R)] -> Double
accuracyBatch outs = let n = (fromIntegral $ length outs)
                         hits = fromIntegral $ length $ do
                           a <- outs
                           let (idx, idx') = bimap (fmap maxIndex . toRows) (fmap maxIndex . toRows) a
                           filter id (zipWith (==) idx idx')
                     in  hits/n
  
