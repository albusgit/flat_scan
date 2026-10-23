package com.albus.flatscan.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.dp

/** The app's own line icons (24 dp, 1.8 stroke like the Material outlined set); Icon tints them. */
object PaperIcons {
    /** The Flatten tool: an iron on the page, heat lines under it. */
    val Iron: ImageVector by lazy { iron(lifted = false) }

    /** The Flatten tool turned off: the iron lifted off the board. */
    val IronLifted: ImageVector by lazy { iron(lifted = true) }

    /** Export: a folded paper airplane. */
    val PaperPlane: ImageVector by lazy {
        ImageVector.Builder("PaperPlane", 24.dp, 24.dp, 24f, 24f).apply {
            stroke("M3 10.5 21 3l-6.5 18-3.8-7.2z M10.7 13.8 21 3")
        }.build()
    }

    private const val IRON_BODY = "M2.5 17.5 6 9.6C6.8 7.9 8.4 7 10.2 7H20c1 0 1.8.8 1.8 1.8v8.7z " +
        "M11.5 7c0-1.9 1-3 2.8-3h4.4c.7 0 1.2.5 1.2 1.2V7"
    private const val IRON_DIAL = "M15.3 12a1.2 1.2 0 1 0 2.4 0a1.2 1.2 0 1 0 -2.4 0z"

    private fun iron(lifted: Boolean) = ImageVector.Builder(if (lifted) "IronLifted" else "Iron", 24.dp, 24.dp, 24f, 24f).apply {
        if (lifted) {
            addGroup(translationY = -2.5f)
            stroke(IRON_BODY)
            fill(IRON_DIAL)
            clearGroup()
            stroke("M3 21h18")
        } else {
            stroke(IRON_BODY)
            fill(IRON_DIAL)
            stroke("M6 20.5h1.5M11 20.5h1.5M16 20.5h1.5")
        }
    }.build()

    private fun ImageVector.Builder.stroke(d: String) = addPath(
        addPathNodes(d), stroke = SolidColor(Color.Black), strokeLineWidth = 1.8f,
        strokeLineCap = StrokeCap.Round, strokeLineJoin = StrokeJoin.Round,
    )

    private fun ImageVector.Builder.fill(d: String) = addPath(addPathNodes(d), fill = SolidColor(Color.Black))
}
