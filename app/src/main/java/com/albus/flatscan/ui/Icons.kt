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

    /** The Filters tab: three overlapping lenses. */
    val Filters: ImageVector by lazy {
        ImageVector.Builder("Filters", 24.dp, 24.dp, 24f, 24f).apply {
            stroke(circle(9f, 9.5f, 5.5f))
            stroke(circle(15f, 9.5f, 5.5f))
            stroke(circle(12f, 14.5f, 5.5f))
        }.build()
    }

    /** The Shape tab: a sheet whose curled top edge lies flat lower down. */
    val PageShape: ImageVector by lazy {
        ImageVector.Builder("PageShape", 24.dp, 24.dp, 24f, 24f).apply {
            stroke("M4 8c2.7-2.5 5.3 1.5 8-1s5.3-1.5 8 0V20H4z")
            stroke("M8 13h8M8 16.5h8")
        }.build()
    }

    /** The Clean up tab: two sparkles. */
    val Sparkles: ImageVector by lazy {
        ImageVector.Builder("Sparkles", 24.dp, 24.dp, 24f, 24f).apply {
            stroke("M10 3.5l1.6 4.4 4.4 1.6-4.4 1.6L10 15.5l-1.6-4.4L4 9.5l4.4-1.6z")
            stroke("M17.5 13.5l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z")
        }.build()
    }

    /** Export: an arrow out of a tray. */
    val Export: ImageVector by lazy {
        ImageVector.Builder("Export", 24.dp, 24.dp, 24f, 24f).apply {
            stroke("M12 15V4M7.5 8.5 12 4l4.5 4.5M5 13v6h14v-6")
        }.build()
    }

    /** Kinds of page: a sheet, a board on its stand, a monitor. */
    val KindDocument: ImageVector by lazy {
        ImageVector.Builder("KindDocument", 24.dp, 24.dp, 24f, 24f).apply { stroke("M6 2.5h8l4.5 4.5v14.5H6z M14 2.5V7h4.5") }.build()
    }
    val KindBoard: ImageVector by lazy {
        ImageVector.Builder("KindBoard", 24.dp, 24.dp, 24f, 24f).apply { stroke("M4 3.5h16a1.5 1.5 0 0 1 1.5 1.5v10a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 15V5A1.5 1.5 0 0 1 4 3.5z M8 21l4-4.5 4 4.5") }.build()
    }
    val KindScreen: ImageVector by lazy {
        ImageVector.Builder("KindScreen", 24.dp, 24.dp, 24f, 24f).apply { stroke("M4 4h16a1.5 1.5 0 0 1 1.5 1.5V15a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 15V5.5A1.5 1.5 0 0 1 4 4z M8.5 20.5h7M12 16.5v4") }.build()
    }

    private fun circle(cx: Float, cy: Float, r: Float) = "M${cx - r} ${cy}a$r $r 0 1 0 ${2 * r} 0a$r $r 0 1 0 ${-2 * r} 0z"

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
