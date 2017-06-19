-- tests/Data/NetworkSpec.hs
module Data.NetworkSpec (spec) where

import Test.Hspec

import Data.Network as N

spec :: Spec
spec = do
  describe "Layers datatype" $ do
    describe "instance IsList (Layers a)" $ do
      describe "toList" $ do
        it "Converts a (Layers a) to a [a]" $ do
          N.toList (1 :=> 2 :=> (Output 3)) `shouldBe` [1,2,3]
      describe "fromList" $ do
        it "Creates a (Layers a) instance from a list of a" $ do
          N.fromList [1,2,3] `shouldBe` (1 :=> 2 :=> (Output 3))
