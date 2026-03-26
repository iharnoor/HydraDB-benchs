"""
Build an enriched PDF of Harnoor & Katie's relationship timeline.
Each photo entry has rich context for temporal reasoning benchmarks.
"""

from fpdf import FPDF
from PIL import Image
import os

BASE = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE, "images")
OUTPUT = os.path.join(BASE, "relationship_timeline.pdf")


# ── Timeline entries in chronological order ──────────────────────
ENTRIES = [
    {
        "date": "October 18, 2020",
        "title": "Meeting for the Second Time",
        "image": "image18.jpeg",
        "description": (
            "Harnoor and Katie met for the second time on October 18, 2020. "
            "They had first connected earlier that month through mutual friends at Florida State University. "
            "This was still very early in their relationship — they were just getting to know each other. "
            "They spent the evening hanging out outdoors, lying on the grass and taking a playful upside-down selfie together. "
            "It was a casual, lighthearted meetup that marked the real beginning of their story. "
            "At this point, neither of them had a car, so getting around Tallahassee required planning."
        ),
    },
    {
        "date": "November 2, 2020",
        "title": "First Car Date",
        "image": "image7.jpeg",
        "description": (
            "On November 2, 2020, Harnoor and Katie went on their first car date. "
            "Katie was driving, and Harnoor sat in the passenger seat. This was a meaningful moment "
            "because it was one of their first real dates outside of campus. They drove around "
            "Tallahassee, exploring the area together. The Synovus bank building is visible through "
            "the windshield. Katie was a confident driver and Harnoor was clearly enjoying the ride. "
            "This was about two weeks after they met for the second time, showing how quickly "
            "they were spending more time together. They were both FSU students at the time."
        ),
    },
    {
        "date": "December 23, 2020",
        "title": "Harnoor's Birthday Celebration",
        "image": "image6.jpeg",
        "description": (
            "Katie surprised Harnoor with a birthday celebration on December 23, 2020. "
            "This was their first birthday together as a couple — they had only been dating for about "
            "two months at this point. Katie got him a large blue 'Happy Birthday' gift bag. "
            "They celebrated at what appears to be Harnoor's apartment in Tallahassee. "
            "Harnoor was wearing his signature leather jacket and glasses, and looked genuinely surprised and happy. "
            "This birthday marked an important early milestone — Katie putting effort into making "
            "the day special showed the relationship was becoming serious. It was also winter break "
            "at FSU, so the fact that they stayed in Tallahassee to celebrate together was meaningful."
        ),
    },
    {
        "date": "January 1, 2021",
        "title": "Traveling to Miami — New Year's Trip",
        "image": "image3.jpeg",
        "description": (
            "On January 1, 2021, Harnoor and Katie traveled to Miami together to ring in the New Year. "
            "This photo was taken at the airport while waiting for their flight. They were sitting by a large "
            "American flag mural in the terminal, with Katie working on a laptop. "
            "This was their first trip together as a couple — a huge relationship milestone. "
            "They had been dating for about two and a half months at this point. "
            "Harnoor was wearing a white T-shirt with a graphic print and his glasses, "
            "while Katie had her hair down. The airport was busy with other New Year travelers. "
            "They were both excited about their first vacation together, heading to South Beach."
        ),
    },
    {
        "date": "February 14, 2021",
        "title": "First Valentine's Day Dinner",
        "image": "image8.jpeg",
        "description": (
            "Harnoor and Katie celebrated their first Valentine's Day together on February 14, 2021. "
            "They went to an upscale restaurant in Tallahassee for a romantic dinner date. "
            "Both were dressed up nicely — Katie wore an elegant navy off-shoulder dress, and "
            "Harnoor wore a dark blazer over a black shirt. They were seated at a wooden table "
            "with wine glasses and water. The restaurant had a warm ambiance with artwork on the walls "
            "and a colorful decorative piece nearby. This was about four months into their relationship, "
            "and it was the first time they did a formal, dressed-up date night. "
            "It represented a shift from casual college hangouts to more intentional, romantic dates."
        ),
    },
    {
        "date": "March 16, 2021",
        "title": "Katie's Birthday",
        "image": "image17.jpeg",
        "description": (
            "Harnoor celebrated Katie's birthday on March 16, 2021. He got her a beautiful white "
            "layered birthday cake that read 'Happy Birthday Katie' with chocolate and cream decorations. "
            "They celebrated at what appears to be Katie's family home or a friend's house — a warm kitchen "
            "setting with wooden cabinets. Harnoor was wearing his glasses and a black hoodie, and both were "
            "beaming with happiness behind the cake. This was significant because it was the first time "
            "Harnoor celebrated Katie's birthday, reciprocating the effort she had made for his birthday "
            "back in December 2020. They had now been together for about five months. "
            "The homemade celebration feel showed their comfort with each other's personal spaces."
        ),
    },
    {
        "date": "April 19, 2021",
        "title": "Revisiting the Place of Their First Date",
        "image": "image15.jpeg",
        "description": (
            "On April 19, 2021, Harnoor and Katie went back to the location of their very first date. "
            "They stood on a stone bridge overlooking a scenic river or lake surrounded by lush green trees. "
            "Harnoor wore a maroon FSU T-shirt and a light blue jacket, while Katie wore a green striped top. "
            "Revisiting their first date spot about six months later was a nostalgic and romantic gesture. "
            "The scenery was beautiful — it was spring in Florida with everything in full bloom. "
            "This trip back to where it all started showed that they were reflecting on how far "
            "they had come as a couple. The location appears to be a park or nature area near Tallahassee "
            "that held special sentimental value for both of them."
        ),
    },
    {
        "date": "May 30, 2021",
        "title": "First Beach Trip to FSU",
        "image": "image1.jpeg",
        "description": (
            "On May 30, 2021, Harnoor and Katie took a photo together in front of the iconic "
            "Westcott Building and fountain at Florida State University. This was described as their "
            "'first beach trip to FSU' — likely combining a visit to a nearby Florida beach with a stop "
            "at their university campus. Harnoor was wearing casual shorts and a T-shirt with a backpack, "
            "while Katie wore a striped top. The Westcott Building with its distinctive red brick architecture "
            "and twin towers is the most recognizable landmark at FSU. The fountain was flowing on a "
            "beautiful sunny day with palm trees and blue skies. This was about seven months into their "
            "relationship, and they were now well-established as a couple on campus. "
            "FSU held deep meaning for them as the place where they first met."
        ),
    },
    {
        "date": "July 3, 2021",
        "title": "Visiting Princeton University",
        "image": "image14.jpeg",
        "description": (
            "On July 3, 2021, Harnoor and Katie visited Princeton University in New Jersey. "
            "They took a selfie on the Princeton campus with its iconic Gothic stone buildings "
            "and manicured green lawns in the background. They were accompanied by a friend — "
            "a young woman in a red top. This was a summer road trip or vacation, showing they were "
            "now traveling beyond Florida together. Harnoor wore a blue T-shirt and a rain jacket, "
            "and Katie was in a black top. The sky was overcast. Visiting an Ivy League campus together "
            "during the summer suggested they were exploring the East Coast, possibly visiting friends "
            "or family in the Northeast. This was their first trip together outside of Florida, "
            "about eight months into the relationship."
        ),
    },
    {
        "date": "July 4, 2021",
        "title": "Fourth of July Fireworks Celebration",
        "image": "image5.jpeg",
        "description": (
            "Harnoor and Katie celebrated the Fourth of July 2021 together, watching fireworks "
            "in a large crowd. The night sky was lit up with fireworks behind them as they took a selfie. "
            "Harnoor wore a blue polo shirt and Katie was smiling brightly. The crowd around them was "
            "large and festive, with many people holding up phones to capture the fireworks. "
            "This appears to have been a major public fireworks display, possibly in New Jersey or "
            "the East Coast area since they were visiting Princeton the day before. "
            "Celebrating America's Independence Day together was especially meaningful for Harnoor "
            "as an international student from India experiencing this American tradition with Katie. "
            "This was one of several holidays they spent together in 2021."
        ),
    },
    {
        "date": "August 21, 2021",
        "title": "Attending Their First American Wedding Together",
        "image": "image12.jpeg",
        "description": (
            "On August 21, 2021, Harnoor and Katie attended their first American wedding together. "
            "The wedding was at a beautiful outdoor venue with rose gardens, mountain views, and elegant "
            "landscaping. Katie wore a stunning navy blue polka-dot dress, and Harnoor was in a sharp "
            "navy blue suit. They looked like a picture-perfect couple. This was a significant cultural "
            "milestone for Harnoor — attending an American wedding for the first time. Coming from India, "
            "where weddings are multi-day colorful celebrations, experiencing an American outdoor garden "
            "wedding was a new and exciting experience. Going to a wedding together as a couple also "
            "signaled a deeper level of commitment — they were now being invited to important life events "
            "as a unit. They had been together for about ten months at this point."
        ),
    },
    {
        "date": "October 15, 2021",
        "title": "Celebrating Navratri Together",
        "image": "image16.jpeg",
        "description": (
            "On October 15, 2021, Harnoor and Katie celebrated Navratri together — a major Hindu festival. "
            "They both dressed in traditional Indian attire and held dandiya sticks for the Garba dance. "
            "Katie wore a beautiful red and green lehenga with traditional jewelry, fully embracing Indian culture. "
            "Harnoor wore a teal kurta. A banner behind them reads 'HOL RAJ' (likely part of a larger festival banner). "
            "This was incredibly significant — Katie, who is American, fully participated in Harnoor's Indian "
            "cultural traditions. She didn't just attend; she dressed up in full traditional attire and danced Garba. "
            "This showed deep respect and enthusiasm for Harnoor's heritage. For Harnoor, seeing his girlfriend "
            "embrace his culture must have been deeply meaningful. They had been together for about a year. "
            "This was likely organized by the Indian student community at FSU or in the Tallahassee area."
        ),
    },
    {
        "date": "November 25, 2021",
        "title": "Thanksgiving with Katie's Family",
        "image": "image11.jpeg",
        "description": (
            "Harnoor spent Thanksgiving with Katie's family on November 25, 2021. "
            "They sat around a dining table with a traditional Thanksgiving spread — turkey, ham, "
            "cranberries, sides, and other dishes. Katie's father and sister (or another family member) "
            "were also at the table. This was a major relationship milestone — meeting and spending a "
            "holiday with the partner's family. For Harnoor, as an international student from India, "
            "Thanksgiving was not a holiday he grew up with, so experiencing it with an American family "
            "was culturally significant. The warm, homey setting with natural light coming through the windows "
            "showed a comfortable family gathering. Katie and Harnoor had now been together for over a year, "
            "and being included in family holidays showed strong acceptance from Katie's side of the family."
        ),
    },
    {
        "date": "December 25, 2021",
        "title": "Christmas Together",
        "image": "image4.jpeg",
        "description": (
            "Harnoor and Katie celebrated Christmas together on December 25, 2021. "
            "They posed by a decorated Christmas tree with lights and ornaments. Harnoor wore a gray jacket "
            "over a red shirt, and Katie wore a white top with jeans. This was their second Christmas together "
            "(the first being in 2020, early in the relationship), but this time they had been together for "
            "over a year and were much more established as a couple. The cozy indoor setting with holiday "
            "decorations — garland along the wall, the lit tree — showed a warm holiday celebration. "
            "For Harnoor, who grew up celebrating Diwali and other Indian festivals, Christmas with Katie "
            "represented their ongoing cultural exchange. Katie celebrated Navratri with him in October; "
            "now he was celebrating Christmas with her in December. This mutual cultural sharing "
            "was a defining feature of their relationship."
        ),
    },
    {
        "date": "January 1, 2022",
        "title": "New Year's in Miami — South Beach",
        "image": "image2.jpeg",
        "description": (
            "Harnoor and Katie rang in the New Year of 2022 on South Beach in Miami. "
            "They took a selfie on Ocean Drive with the iconic Art Deco buildings and palm trees behind them. "
            "Katie wore a turquoise 'Royal Beach' tank top, and Harnoor was in a casual outfit. "
            "This was their second New Year's trip to Miami — they had also traveled there on January 1, 2021, "
            "exactly one year earlier. Making it an annual tradition showed the consistency and growth of their "
            "relationship. Comparing the two Miami trips: in 2021 they were a brand new couple (2.5 months), "
            "and now in 2022 they were a seasoned couple (over 14 months together). "
            "The Boulevard Hotel is visible in the background. The weather was sunny and warm, "
            "typical of a Miami winter day."
        ),
    },
    {
        "date": "March 3, 2022",
        "title": "Packing for India Trip",
        "image": "image13.jpeg",
        "description": (
            "On March 3, 2022, Harnoor and Katie packed their suitcases for a trip to India. "
            "They posed in their apartment with multiple large suitcases — blue and gray — ready for the journey. "
            "This was a monumental milestone in their relationship. Katie was about to visit India for the first "
            "time, and more importantly, she was going to meet Harnoor's family. For an intercultural couple, "
            "the 'meeting the parents' trip to the other country is one of the biggest steps. "
            "They had been together for about a year and a half at this point. The apartment looked like a "
            "typical college/post-college setup with a kitchen counter visible. The amount of luggage suggested "
            "a long trip — likely several weeks in India. Harnoor was wearing his glasses and a green striped shirt, "
            "while Katie wore a plaid flannel. They both looked excited and a bit nervous for the big trip."
        ),
    },
    {
        "date": "March 2022",
        "title": "Visiting the Taj Mahal in India",
        "image": "image9.jpeg",
        "description": (
            "During their India trip in March 2022, Harnoor and Katie visited the Taj Mahal in Agra. "
            "Katie wore a beautiful turquoise traditional Indian salwar kameez, and Harnoor wore a light blue "
            "button-down shirt. The iconic white marble Taj Mahal is visible in the background with its famous "
            "gardens and reflecting pools. This was a dream-come-true moment — visiting one of the Seven Wonders "
            "of the World together. For Katie, this was her first time in India and seeing the Taj Mahal in person. "
            "For Harnoor, showing Katie his home country and its most famous landmark was deeply personal. "
            "Katie fully embraced wearing Indian clothing during the trip, just as she had done for Navratri "
            "back in October 2021. The trip to India represented the deepest level of cultural immersion in their "
            "relationship — Katie wasn't just learning about Indian culture from afar, she was living it. "
            "They also met Harnoor's family during this trip, who warmly welcomed Katie."
        ),
    },
    {
        "date": "May 7, 2022",
        "title": "Buying a Car Together — Tesla Model 3",
        "image": "image10.JPG",
        "description": (
            "On May 7, 2022, Harnoor and Katie bought a car together — a white Tesla Model 3. "
            "They posed at the Tesla dealership with a Tesla employee who was handing over the keys. "
            "Harnoor wore a gray blazer and Katie stood beside him. This was one of the biggest financial "
            "and practical commitments they made as a couple. Buying a car together signified long-term planning "
            "and shared responsibility. They had been together for about a year and seven months. "
            "Choosing a Tesla showed their interest in technology and sustainability. "
            "Looking back at the timeline: in November 2020 they went on their 'first car date' in someone else's car, "
            "and now in May 2022 they were buying their own car together. That progression from borrowing a car "
            "to co-owning a Tesla captured the growth of their relationship. "
            "The Tesla was white, and the dealership parking lot had charging stations visible in the background."
        ),
    },
]


def sanitize(text):
    """Replace Unicode characters that Helvetica can't handle."""
    return (text
        .replace("\u2014", " - ")   # em dash
        .replace("\u2013", " - ")   # en dash
        .replace("\u2018", "'")     # left single quote
        .replace("\u2019", "'")     # right single quote
        .replace("\u201c", '"')     # left double quote
        .replace("\u201d", '"')     # right double quote
    )


def build_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title Page ──────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 40, "", ln=True)
    pdf.cell(0, 15, "Harnoor & Katie", ln=True, align="C")
    pdf.set_font("Helvetica", "", 18)
    pdf.cell(0, 12, "A Relationship Timeline", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 12, "October 2020 - May 2022", ln=True, align="C")
    pdf.cell(0, 30, "", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, sanitize(
        "This document chronicles the relationship journey of Harnoor Singh and Katie, "
        "from their second meeting in October 2020 through buying a car together in May 2022. "
        "It covers 18 key moments across 19 months, including holidays, cultural exchanges, "
        "family introductions, travel milestones, and shared firsts. "
        "The timeline captures how an intercultural couple - Harnoor from India and Katie from "
        "the United States - built a relationship that bridged two cultures, with both partners "
        "actively embracing each other's traditions and families."
    ), align="C")

    # ── Timeline Entries ────────────────────────────────────
    for entry in ENTRIES:
        pdf.add_page()

        # Date & Title
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, sanitize(entry["date"]), ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, sanitize(entry["title"]), ln=True)
        pdf.cell(0, 4, "", ln=True)

        # Image
        img_path = os.path.join(IMG_DIR, entry["image"])
        if os.path.exists(img_path):
            # Resize to fit page width while maintaining aspect ratio
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    # Fix orientation using EXIF data
                    try:
                        from PIL import ExifTags
                        exif = img._getexif()
                        if exif:
                            for tag, value in exif.items():
                                if ExifTags.TAGS.get(tag) == "Orientation":
                                    if value == 3:
                                        img = img.rotate(180, expand=True)
                                    elif value == 6:
                                        img = img.rotate(270, expand=True)
                                    elif value == 8:
                                        img = img.rotate(90, expand=True)
                                    w, h = img.size
                                    # Save corrected image
                                    corrected = img_path + ".corrected.jpg"
                                    img.save(corrected, "JPEG", quality=85)
                                    img_path = corrected
                    except Exception:
                        pass

                max_w = 170  # mm
                max_h = 100  # mm
                aspect = w / h
                if aspect > max_w / max_h:
                    img_w = max_w
                    img_h = max_w / aspect
                else:
                    img_h = max_h
                    img_w = max_h * aspect

                x = (210 - img_w) / 2  # center on A4
                pdf.image(img_path, x=x, w=img_w, h=img_h)
                pdf.cell(0, 5, "", ln=True)
            except Exception as e:
                pdf.cell(0, 10, f"[Image: {entry['image']}]", ln=True)

        # Description
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, sanitize(entry["description"]))

    pdf.output(OUTPUT)
    print(f"PDF created: {OUTPUT}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    build_pdf()
