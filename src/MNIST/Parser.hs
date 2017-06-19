module Parser (renderImage, parseMNISTSet, Image, Label, MNISTSet, MNISTSample) where

-- import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString as B
import Data.Functor
-- import System.Random
import Data.Word8
import Data.Binary.Get
import Control.Monad
import Control.Monad.Except as E
import qualified Data.Vector.Storable as V
-- import Numeric.LinearAlgebra
-- import Numeric.LinearAlgebra.Devel

data IDXType = IDXUByte | IDXSByte | IDXShort | IDXInt | IDXFloat | IDXLong | IDXWTF deriving (Show)

-- IDX format header :: (datatype, dimensionality, [size of each dimension])
makeIDXType :: Word8 -> Maybe IDXType
makeIDXType 0x08 = Just IDXUByte
makeIDXType 0x09 = Just IDXSByte
makeIDXType 0x0B = Just IDXShort
makeIDXType 0x0C = Just IDXInt
makeIDXType 0x0D = Just IDXFloat
makeIDXType 0x0E = Just IDXLong
makeIDXType _    = Nothing

type IDXParser = E.ExceptT String Get

runIDXParser :: IDXParser a -> BS.ByteString -> Either String a
runIDXParser p bs = runGet (runExceptT p) bs


type Header = (IDXType, Int, [Int])

data IDXData = IDXData Header BS.ByteString

instance Show IDXData where
  show (IDXData (t, d, s:ss) _) = show s ++ " " ++ show d ++ "-dimensional samples of type " ++ show t

parseIDXHeader :: IDXParser Header
parseIDXHeader = do
  (0x00 : 0x00 : t : ds : []) <- lift $ replicateM 4 getWord8
  let ds' = fromIntegral ds
  case makeIDXType t of
    Just t' -> do 
      sizes <- lift $ (replicateM ds' getWord32be) 
      return (t', ds', fromIntegral <$> sizes)
    Nothing -> throwError "Invalid byte for datatype"

parseIDXData :: IDXParser IDXData
parseIDXData = do
  h <- parseIDXHeader
  bs <- lift getRemainingLazyByteString
  return $ IDXData h bs

type Image = V.Vector Double
type Label = V.Vector Double

type MNISTSample = (Image, Label)
-- MNISTSet = ((nrows, ncols), nsamples, mnistsamples)
type MNISTSet = ((Int, Int), Int, [MNISTSample])

-- maps a Word8 value into a number between 0 and 1
word8ToDouble :: Word8 -> Double
word8ToDouble w = (fromIntegral w) / 255.0


chunksOf :: V.Storable a => Int -> V.Vector a -> [V.Vector a]
chunksOf n v
  | V.length v == 0 = []
  | n < V.length v = let (c, c') = V.splitAt n v
                     in  c:chunksOf n c'
  | otherwise = [v]

-- Parse a sequence of bytes(Word8) into a vector of numbers(Double)
getImage :: (Int, Int) -> Get Image
getImage (nrows, ncols) = do
  bs <- getByteString (nrows*ncols) 
  return $ V.generate (nrows*ncols) $ \i -> word8ToDouble (B.index bs i)

getImages :: Int -> (Int,Int) -> Get [Image]
getImages n (nrows, ncols) = do
  replicateM n (getImage (nrows, ncols))

parseMNISTLabels :: IDXParser (Int, [Label])
parseMNISTLabels = do
  idxdata <- parseIDXData
  case idxdata of
    IDXData (IDXUByte, 1, [nlabels]) lbs -> return $ (nlabels, fmap word8ToVector $ BS.unpack lbs)
    IDXData (_, 1, [nlabels]) lbs -> throwError "Wrong data type for mnist labels"
    IDXData (_, _, _) lbs -> throwError "Wrong data dimensionality for mnist labels set"
  
parseMNISTImages :: IDXParser (Int, (Int,Int), [Image])
parseMNISTImages = do
  idxdata <- parseIDXData
  case idxdata of
    IDXData (IDXUByte, 3, [nimages, nrows, ncols]) ibs ->
      let images = runGet (getImages nimages (nrows,ncols)) ibs
      in return (nimages, (nrows, ncols), images)
    IDXData (_, 3, [nimages, nrows, ncols]) ibs -> throwError "Wrong data type for mnist images"
    IDXData (_, _, _) ibs -> throwError "Wrong data dimensionality for mnist images set"

parseMNISTSet :: BS.ByteString -> BS.ByteString -> Either String MNISTSet
parseMNISTSet ibs lbs = do
  (nlabels, labels) <- runIDXParser parseMNISTLabels lbs
  (nimages, (nrows, ncols), images) <- runIDXParser parseMNISTImages ibs
  if nlabels == nimages
    then
    let samples = zip images labels
        nsamples = nlabels
    in return ((nrows, ncols), nsamples, samples)
    else throwError "Number of labels not equal to number of images"

renderPixel :: Double -> Char  
renderPixel n = let s = " .:oO@" in s !! (floor (n*256) * length s `div` 256)

renderImage :: (Int, Int) -> Image -> String
renderImage (nrows, ncols) = unlines . (map V.toList) . (chunksOf ncols) . (V.map renderPixel) 

-- converts a Label(a byte representing a digit) to a vector corresponding to the desired ouput of a network 
word8ToVector :: Word8 -> V.Vector Double
word8ToVector l = V.generate 10 $ \i -> if i == (fromIntegral l) then 1 else 0

  
