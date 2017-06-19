-- tests/Data/Network/FullyConnected.hs
module FullyConnectedSpec (spec) where

import Test.Hspec

import Data.Network
import Data.Network.Activation
import Data.Network.FullyConnected as FC
import System.Random
import Numeric.LinearAlgebra

spec :: Spec
spec = do
  describe "A fully connected layer composed of a weight matrix, a bias vector and an activation function" $ do
    describe "instance Layer (FullyConnected (Matrix R) R) where" $ do
      describe "makeRandomLayer" $ do
        it "creates a randomly initialized layer" $ do
          g <- getStdGen
          let l = makeRandomLayer g id 10 20 (logistic :: ActivationFunction (Matrix Double)) :: FullyConnected (Matrix Double) Double
          (size (weights l)) `shouldBe` (10,20)
          (size (biases l)) `shouldBe` 20
          let (g',g'') = split g
          let rs = randoms g'
          let i = (10><20) rs
          (activate (activation l) i) `shouldBe` (activate logistic i)
      describe "compute" $ do
        it "Computes the linear combination of the layer's inputs" $ do
          g <- getStdGen
          let (g1,g2) = split g
          let l@(FC w b af) = makeRandomLayer g1 id 10 20 (logistic) :: FullyConnected (Matrix Double) Double
          let rs = randoms g2
          let i = (100><10) rs
          let o = compute l i
          size o `shouldBe` (100,20)
          o `shouldBe` ((fromRows $ replicate 100 b) + (i <> w))

    describe "instance Network i (Layers (FullyConnected i Double)) where" $ do
      describe "makeRandomNetwork" $ do
        it "Creates a randomly initialized network given the specification of the layers" $ do
          g <- getStdGen
          let (g1,g2) = split g
          let n = makeRandomNetwork g1 id 10 [(30, logistic)] (20, logistic) :: FeedForwardFC (Matrix R)
          pending
          
        
