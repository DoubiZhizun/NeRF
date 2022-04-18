package nerf_change

import ai.djl.ndarray._
import ai.djl.translate._

class nerfTranslator extends Translator[NDList, NDList] {
  override def processInput(ctx: TranslatorContext, input: NDList): NDList = input

  override def processOutput(ctx: TranslatorContext, list: NDList): NDList = list

}