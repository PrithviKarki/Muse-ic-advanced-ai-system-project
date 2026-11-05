"""
Curated Top 100 Most Popular English Songs with Spotify-style audio telemetry
(genre, mood, energy, tempo_bpm, valence, danceability, acousticness), grounded
RAG descriptions, and clickable YouTube Music links.
"""

import urllib.parse
from typing import Any, Dict, List

# Format: (Title, Artist, Genre, Mood, Energy, BPM, Valence, Danceability, Acousticness, Description, Themes)
TOP_100_ENGLISH_RAW: List[tuple] = [
    # --- Pop & Synth-Pop Mega-Hits ---
    ("Blinding Lights", "The Weeknd", "synthwave", "intense", 0.80, 171, 0.33, 0.51, 0.00, "80s-inspired synthwave and electro-pop anthem driven by pulsing analog arpeggios and nocturnal highway urgency.", "night drive, workout, cardio, running, 80s, synthwave, pop, high energy, neon"),
    ("As It Was", "Harry Styles", "indie pop", "nostalgic", 0.73, 174, 0.66, 0.52, 0.34, "Breezy synth-driven indie pop hit pairing propulsive drum-machine beats with bittersweet nostalgic vocals.", "indie pop, upbeat, driving, nostalgic, summer, running, pop"),
    ("Birds of a Feather", "Billie Eilish", "indie pop", "romantic", 0.51, 105, 0.44, 0.75, 0.20, "Lush mid-tempo indie pop song with warm synth pads, breezy acoustic rhythm, and devoted vocal harmonies.", "chill, romantic, indie pop, study, relaxed, evening, breezy"),
    ("Espresso", "Sabrina Carpenter", "pop", "groovy", 0.76, 104, 0.69, 0.70, 0.11, "Infectious disco-funk pop groove with snappy basslines, sunny synths, and confident upbeat bounce.", "pop, groovy, happy, party, summer, upbeat, funk, dance"),
    ("Levitating", "Dua Lipa", "pop", "happy", 0.83, 103, 0.92, 0.70, 0.01, "Nu-disco pop powerhouse featuring slap bass, handclaps, and euphoric dancefloor energy.", "party, dance, workout, cardio, happy, disco, funk, pop"),
    ("Cruel Summer", "Taylor Swift", "pop", "euphoric", 0.70, 170, 0.56, 0.55, 0.12, "Soaring synth-pop anthem with industrial synth bass, dramatic bridge, and high-energy summer intensity.", "pop, summer, upbeat, workout, driving, euphoric, happy"),
    ("Starboy", "The Weeknd & Daft Punk", "r&b", "confident", 0.59, 186, 0.49, 0.68, 0.14, "Sleek electro-R&B collaboration built on vocoder harmonies, crisp piano hooks, and moody sub-bass swagger.", "r&b, night drive, confident, moody, electronic, workout"),
    ("Don't Start Now", "Dua Lipa", "pop", "groovy", 0.79, 124, 0.68, 0.79, 0.01, "Disco-house pop anthem driven by live cowbell, funky bassline slides, and exuberant strings.", "disco, party, dance, workout, cardio, groovy, pop"),
    ("Watermelon Sugar", "Harry Styles", "pop", "happy", 0.82, 95, 0.56, 0.55, 0.12, "Sun-drenched horn-and-guitar pop groove radiating carefree summer beach warmth.", "summer, happy, beach, upbeat, pop,sunny"),
    ("Viva La Vida", "Coldplay", "indie pop", "euphoric", 0.62, 138, 0.42, 0.49, 0.10, "Sweeping orchestral art-pop anthem propelled by staccato strings, timpani bells, and stadium chorus.", "euphoric, uplifting, running, workout, strings, orchestra, indie pop"),
    ("Shape of You", "Ed Sheeran", "pop", "groovy", 0.65, 96, 0.93, 0.83, 0.58, "Tropical-house and acoustic marimba pop groove with percussive loop and high danceability.", "pop, dance, party, groovy, happy, acoustic"),
    ("Someone You Loved", "Lewis Capaldi", "pop", "sorrowful", 0.41, 110, 0.45, 0.50, 0.75, "Raw acoustic piano ballad delivering emotional catharsis and heartache through soaring vocals.", "sad, melancholic, sorrowful, piano, acoustic, heartache, late night"),
    ("Bad Guy", "Billie Eilish", "pop", "moody", 0.43, 135, 0.56, 0.70, 0.33, "Minimalist bass-heavy electro-pop track with whispered vocals and dark quirky groove.", "moody, bass, electronic, dark, pop"),
    ("Sunflower", "Post Malone & Swae Lee", "hip hop", "chill", 0.52, 90, 0.91, 0.76, 0.56, "Warm melodic hip-hop and dream-pop crossover with laid-back drums and bright sunny vocals.", "chill, happy, hip hop, pop, summer, study, relaxed"),
    ("Locked Out of Heaven", "Bruno Mars", "funk", "happy", 0.80, 144, 0.87, 0.73, 0.05, "New-wave reggae-rock and funk-pop explosion with punchy horns and high-octane rhythm.", "funk, pop, party, workout, happy, upbeat, dance"),
    ("Uptown Funk", "Mark Ronson & Bruno Mars", "funk", "festive", 0.61, 115, 0.93, 0.86, 0.01, "80s Minneapolis funk and boogie brass celebration packed with slap bass and party swagger.", "funk, party, dance, festive, groovy, happy"),
    ("Billie Jean", "Michael Jackson", "pop", "groovy", 0.65, 117, 0.85, 0.92, 0.02, "Defining post-disco pop and funk masterpiece anchored by an iconic bassline and crisp drum groove.", "pop, funk, dance, groovy, 80s, party"),
    ("Everybody Wants to Rule the World", "Tears for Fears", "synthwave", "nostalgic", 0.80, 112, 0.54, 0.65, 0.36, "Timeless 1985 new-wave and synth-pop anthem with shuffling rhythm, shimmering guitars, and highway nostalgia.", "80s, synthwave, night drive, nostalgic, pop, highway"),
    ("Take On Me", "a-ha", "synthwave", "happy", 0.90, 169, 0.88, 0.57, 0.01, "High-tempo 80s synth-pop classic propelled by iconic Juno-60 arpeggios and soaring falsetto.", "80s, synthwave, happy, workout, cardio, running, upbeat"),
    ("Africa", "Toto", "pop", "nostalgic", 0.67, 93, 0.73, 0.67, 0.26, "Lush soft-rock and synth-percussion classic featuring CS-80 brass pads, kalimba marimba, and warm harmonies.", "80s, nostalgic, relaxed, pop, soft rock"),

    # --- Iconic Rock & Alternative Anthems ---
    ("Bohemian Rhapsody", "Queen", "rock", "intense", 0.40, 71, 0.22, 0.41, 0.27, "Operatic rock suite shifting from intimate grand piano balladry to explosive hard-rock guitar solos.", "rock, classical, piano, dramatic, intense, classic rock"),
    ("Dreams", "Fleetwood Mac", "rock", "chill", 0.49, 120, 0.79, 0.83, 0.06, "Hypnotic California soft-rock groove pairing warm Fender Rhodes, steady bass heartbeat, and Stevie Nicks vocals.", "chill, relaxed, rock, driving, nostalgic, sunday, coffee"),
    ("Hotel California", "Eagles", "rock", "moody", 0.51, 75, 0.61, 0.58, 0.01, "Cinematic 12-string acoustic and twin-lead electric guitar rock epic about nocturnal desert highways.", "rock, night drive, guitar, moody, classic rock"),
    ("Smells Like Teen Spirit", "Nirvana", "rock", "aggressive", 0.91, 117, 0.72, 0.50, 0.00, "Raw 90s grunge rock explosion driven by loud-quiet-loud distorted guitar riffs and thunderous drums.", "rock, workout, gym, intense, aggressive, heavy, grunge"),
    ("Mr. Brightside", "The Killers", "rock", "intense", 0.93, 148, 0.24, 0.35, 0.00, "High-adrenaline indie rock anthem with rapid arpeggiated guitar, driving 148 BPM drums, and festival energy.", "rock, workout, cardio, running, party, intense, high energy"),
    ("Do I Wanna Know?", "Arctic Monkeys", "rock", "moody", 0.53, 85, 0.42, 0.55, 0.19, "Sultry mid-tempo indie rock groove built on a heavy 12-string guitar riff and late-night stomp.", "rock, night drive, moody, late night, guitar"),
    ("Wonderwall", "Oasis", "rock", "nostalgic", 0.89, 87, 0.65, 0.38, 0.00, "Britpop acoustic-electric singalong anthem with iconic strumming pattern and cello backing.", "rock, acoustic, guitar, nostalgic, singalong"),
    ("Sweater Weather", "The Neighbourhood", "indie pop", "chill", 0.81, 124, 0.40, 0.61, 0.05, "Atmospheric indie-rock and alt-pop crossover blending reverberant guitar, beach-autumn mood, and steady groove.", "indie pop, chill, moody, autumn, night drive, rainy"),
    ("Creep", "Radiohead", "rock", "melancholic", 0.43, 92, 0.10, 0.52, 0.01, "Somber alternative rock ballad punctuated by Johnny Greenwood's crunchy guitar blasts and existential longing.", "melancholic, sad, existential dread, sorrowful, rock, late night, introspection"),
    ("Seven Nation Army", "The White Stripes", "rock", "aggressive", 0.46, 124, 0.32, 0.74, 0.01, "Iconic garage-rock anthem driven by an octave-pedal guitar riff and pounding stadium kick drum.", "rock, workout, gym, intense, aggressive, stadium"),
    ("In the End", "Linkin Park", "rock", "intense", 0.86, 105, 0.40, 0.56, 0.01, "Nu-metal and rap-rock classic contrasting melancholic piano motifs with heavy distorted guitars.", "rock, metal, workout, gym, intense, piano"),
    ("Bring Me To Life", "Evanescence", "metal", "intense", 0.94, 95, 0.29, 0.33, 0.01, "Symphonic gothic metal and hard rock anthem fusing haunting piano with heavy drop-tuned guitar riffs.", "metal, rock, workout, gym, intense, heavy, dramatic"),
    ("Enter Sandman", "Metallica", "metal", "aggressive", 0.84, 123, 0.60, 0.58, 0.00, "Heavy metal staple built on a menacing acoustic-to-electric riff and thunderous powerlifting percussion.", "metal, heavy, powerlifting, gym, workout, aggressive, intense"),
    ("Back In Black", "AC/DC", "rock", "confident", 0.70, 94, 0.51, 0.31, 0.01, "Hard-hitting stadium rock classic powered by crunchy blues-rock guitar chords and swaggering rhythm.", "rock, workout, gym, confident, guitar"),
    ("Sweet Child O' Mine", "Guns N' Roses", "rock", "euphoric", 0.90, 125, 0.67, 0.45, 0.09, "Soaring hard rock anthem featuring Slash's iconic melodic guitar intro and high-energy solo.", "rock, guitar, workout, upbeat, high energy"),
    ("Every Breath You Take", "The Police", "rock", "moody", 0.45, 117, 0.74, 0.82, 0.54, "Minimalist new-wave rock classic built on Andy Summers' palm-muted arpeggiated guitar motif.", "rock, 80s, moody, chill, guitar"),
    ("Under the Bridge", "Red Hot Chili Peppers", "rock", "melancholic", 0.35, 85, 0.46, 0.56, 0.11, "Intimate fingerpicked electric guitar ballad about solitude and reflection in Los Angeles.", "rock, melancholic, guitar, chill, introspection"),
    ("Californication", "Red Hot Chili Peppers", "rock", "chill", 0.77, 96, 0.33, 0.59, 0.00, "Melodic alternative rock track featuring clean interplay between Stratocaster guitar and melodic bass.", "rock, guitar, driving, chill"),
    ("Viva La Gloria / Boulevard of Broken Dreams", "Green Day", "rock", "melancholic", 0.68, 83, 0.47, 0.50, 0.01, "Tremolo-effected guitar rock ballad walking through lonely midnight streets.", "rock, melancholic, lonely, night, guitar"),
    ("Clocks", "Coldplay", "indie pop", "focused", 0.75, 131, 0.26, 0.58, 0.60, "Propulsive piano-rock anthem driven by an hypnotic ostinato piano arpeggio and atmospheric synth pads.", "piano, focused, coding, study, indie pop, rock"),

    # --- R&B, Soul, Jazz & Lounge Classics ---
    ("Pink + White", "Frank Ocean", "r&b", "peaceful", 0.55, 160, 0.55, 0.50, 0.67, "Sunlit neo-soul and acoustic R&B arrangement featuring piano, acoustic guitar, string swells, and Beyoncé backing vocals.", "r&b, peaceful, chill, acoustic, piano, coffee, sunday, study"),
    ("Kill Bill", "SZA", "r&b", "chill", 0.73, 89, 0.42, 0.64, 0.05, "Psychedelic soul and vintage R&B groove with melodic bass, flute-like synths, and laid-back rhythm.", "r&b, chill, relaxed, groove, evening"),
    ("Snooze", "SZA", "r&b", "romantic", 0.55, 143, 0.39, 0.56, 0.14, "Warm slow-jam R&B track with shimmering guitar chords, lush harmonies, and late-night intimacy.", "r&b, romantic, late night, chill, relaxed"),
    ("Golden Hour", "JVKE", "classical", "romantic", 0.44, 94, 0.15, 0.52, 0.84, "Virtuosic classical-pop crossover driven by sweeping solo piano arpeggios and cinematic orchestral strings.", "classical, piano, acoustic, romantic, study, coding, peaceful"),
    ("Die With A Smile", "Lady Gaga & Bruno Mars", "r&b", "romantic", 0.59, 158, 0.53, 0.52, 0.31, "70s soft-rock and soul ballad featuring warm electric guitar, vintage Rhodes piano, and powerhouse duet vocals.", "romantic, soul, r&b, relaxed, evening"),
    ("Redbone", "Childish Gambino", "funk", "moody", 0.35, 160, 0.59, 0.74, 0.17, "70s psychedelic funk and slow-jam R&B track with Fender Rhodes, envelope-filter bass, and midnight atmosphere.", "funk, r&b, late night, moody, chill, rhodes"),
    ("Ain't No Mountain High Enough", "Marvin Gaye & Tammi Terrell", "r&b", "happy", 0.63, 129, 0.80, 0.66, 0.45, "Motown soul classic bursting with tambourine, upright bass, orchestral strings, and joyful energy.", "soul, r&b, happy, upbeat, feel-good"),
    ("Superstition", "Stevie Wonder", "funk", "groovy", 0.82, 101, 0.87, 0.63, 0.04, "Foundational funk groove built on multi-layered Hohner Clavinet riffs and brass horn stabs.", "funk, groovy, party, dance, upbeat"),
    ("Fly Me To The Moon", "Frank Sinatra & Count Basie", "jazz", "relaxed", 0.46, 119, 0.59, 0.65, 0.72, "Swinging acoustic jazz standard with Count Basie's big-band horns, upright bass, and coffee-lounge elegance.", "jazz, coffee, cafe, lounge, relaxed, acoustic, sunday"),
    ("What a Wonderful World", "Louis Armstrong", "jazz", "peaceful", 0.15, 68, 0.27, 0.29, 0.88, "Timeless acoustic jazz-orchestral ballad radiating serenity, gentle strings, and peaceful reflection.", "jazz, peaceful, calm, acoustic, strings, sleep, sunday"),
    ("Feeling Good", "Nina Simone", "jazz", "confident", 0.42, 78, 0.54, 0.52, 0.69, "Dramatic vocal jazz and blues arrangement opening a cappella before erupting into bold brass and piano.", "jazz, blues, confident, piano, brass"),
    ("Take Five", "The Dave Brubeck Quartet", "jazz", "focused", 0.26, 174, 0.53, 0.45, 0.86, "Cool-jazz instrumental classic in 5/4 time featuring Paul Desmond's silky alto saxophone and acoustic piano.", "jazz, instrumental, study, coding, focus, coffee, saxophone, piano, acoustic"),
    ("Blue in Green", "Miles Davis & Bill Evans", "jazz", "melancholic", 0.05, 62, 0.18, 0.36, 0.94, "Intimate modal jazz ballad from Kind of Blue featuring muted trumpet and Bill Evans' impressionistic felt piano.", "jazz, lofi, study, coding, rain, rainy night, piano, acoustic, melancholic, peaceful"),
    ("At Last", "Etta James", "blues", "romantic", 0.31, 87, 0.48, 0.27, 0.83, "Lush orchestral blues and soul ballad with sweeping strings, acoustic piano, and heartfelt romance.", "blues, romantic, relaxed, strings, piano, acoustic"),
    ("The Thrill Is Gone", "B.B. King", "blues", "sorrowful", 0.51, 90, 0.55, 0.55, 0.42, "Definitive minor-key electric blues anthem pairing weeping Lucille guitar bends with somber string arrangements.", "blues, sorrowful, melancholic, existential dread, heartache, guitar, late night"),
    ("Gravity", "John Mayer", "blues", "peaceful", 0.32, 62, 0.26, 0.59, 0.72, "Slow-burning blues-soul ballad with warm clean Stratocaster guitar, Hammond organ, and contemplative calm.", "blues, guitar, peaceful, chill, relaxed, acoustic, late night"),

    # --- Hip-Hop & Urban Grooves ---
    ("Lose Yourself", "Eminem", "hip hop", "intense", 0.74, 171, 0.06, 0.69, 0.01, "High-stakes motivational hip-hop anthem driven by a tense piano-and-guitar loop and relentless lyrical cadence.", "hip hop, workout, gym, running, intense, motivation, confident"),
    ("HUMBLE.", "Kendrick Lamar", "hip hop", "confident", 0.62, 150, 0.42, 0.91, 0.00, "Hard-hitting West Coast hip-hop track anchored by a staccato piano riff and booming 808 bass.", "hip hop, workout, gym, confident,swagger, 808"),
    ("Luther", "Kendrick Lamar & SZA", "hip hop", "romantic", 0.48, 138, 0.58, 0.72, 0.46, "Soulful hip-hop and R&B duet sampling Luther Vandross with warm acoustic strings and smooth 808 rhythm.", "hip hop, r&b, romantic, chill, relaxed, smooth"),
    ("Stronger", "Kanye West", "hip hop", "intense", 0.72, 104, 0.49, 0.62, 0.01, "Electro-hop stadium workout anthem sampling Daft Punk synths with heavy electronic drums.", "hip hop, edm, workout, gym, intense, electronic, power"),
    ("Empire State of Mind", "JAY-Z & Alicia Keys", "hip hop", "euphoric", 0.95, 173, 0.81, 0.49, 0.03, "Triumphant piano-driven hip-hop and soul anthem celebrating ambition and city lights.", "hip hop, piano, euphoric, confident, upbeat"),
    ("See You Again", "Wiz Khalifa & Charlie Puth", "hip hop", "nostalgic", 0.48, 80, 0.28, 0.69, 0.37, "Emotional piano ballad and hip-hop tribute about friendship, memory, and farewell.", "nostalgic, melancholic, piano, hip hop, pop"),
    ("God's Plan", "Drake", "hip hop", "groovy", 0.45, 77, 0.36, 0.75, 0.03, "Atmospheric trap-pop groove with bouncy 808 bass and laid-back melodic flow.", "hip hop, groovy, chill, confident"),
    ("Good Days", "SZA", "r&b", "peaceful", 0.43, 121, 0.41, 0.46, 0.49, "Dreamy acoustic-guitar R&B meditation on letting go of anxiety and finding inner peace.", "r&b, peaceful, chill, acoustic, guitar, meditation, calm"),

    # --- Electronic, Synthwave, EDM & Indie Dance ---
    ("Get Lucky", "Daft Punk & Pharrell Williams", "funk", "groovy", 0.81, 116, 0.86, 0.79, 0.04, "Live-recorded disco-funk masterpiece featuring Nile Rodgers' rhythm guitar, vocoders, and warm analog groove.", "funk, disco, groovy, happy, party, dance, night drive"),
    ("Instant Crush", "Daft Punk & Julian Casablancas", "synthwave", "melancholic", 0.59, 110, 0.52, 0.78, 0.04, "Nostalgic synth-rock ballad blending vocoder melodies, analog synthesizers, and wistful guitar solos.", "synthwave, nostalgic, melancholic, night drive, indie pop"),
    ("Midnight City", "M83", "synthwave", "euphoric", 0.71, 105, 0.29, 0.53, 0.02, "Defining neon synthwave anthem with iconic synth lead hook and explosive late-night saxophone outro.", "synthwave, night drive, neon, 80s, euphoric, highway, saxophone"),
    ("Nightcall", "Kavinsky", "synthwave", "moody", 0.58, 91, 0.36, 0.58, 0.02, "Dark outrun synthwave classic crafted for midnight highway driving with robotic vocoders and pulsing bass.", "synthwave, cyberpunk, night drive, moody, dark, 80s, neon"),
    ("The Less I Know The Better", "Tame Impala", "indie pop", "groovy", 0.74, 117, 0.78, 0.64, 0.01, "Psychedelic disco-rock groove anchored by an infectious fuzz-bass riff and vintage clavinet synths.", "indie pop, groovy, funk, party, upbeat"),
    ("Wake Me Up", "Avicii", "edm", "euphoric", 0.78, 124, 0.64, 0.53, 0.00, "Festival progressive-house and folk-EDM crossover pairing acoustic guitar strumming with euphoric synth drops.", "edm, workout, cardio, running, festival, euphoric, happy"),
    ("Titanium", "David Guetta & Sia", "edm", "intense", 0.79, 126, 0.30, 0.60, 0.07, "Explosive electro-house anthem with soaring vocal power and high-impact festival synth drops.", "edm, workout, gym, cardio, intense, power"),
    ("Levels", "Avicii", "edm", "euphoric", 0.89, 126, 0.60, 0.58, 0.00, "Iconic progressive-house raver built on an Etta James vocal sample and infectious piano-synth hook.", "edm, rave, festival, party, workout, euphoric"),
    ("One More Time", "Daft Punk", "edm", "festive", 0.70, 123, 0.48, 0.61, 0.02, "French-house celebration with filtered disco horns and uplifting robotic vocals.", "edm, disco, party, dance, festive, happy"),
    ("Closer", "The Chainsmokers & Halsey", "edm", "nostalgic", 0.52, 95, 0.66, 0.75, 0.41, "Melodic future-bass pop hit with nostalgic lyrics and warm synth chords.", "edm, pop, nostalgic, chill"),

    # --- Folk, Indie Acoustic, Rainy Study & Introspection ---
    ("Too Sweet", "Hozier", "folk", "groovy", 0.62, 117, 0.68, 0.74, 0.32, "Bluesy folk-soul groove about late-night whisky and black coffee with warm bass and gospel organ.", "folk, blues, coffee, late night, groovy, chill"),
    ("Take Me To Church", "Hozier", "folk", "intense", 0.66, 129, 0.44, 0.57, 0.63, "Passionate gospel-folk ballad driven by resonant piano chords and raw vocal intensity.", "folk, piano, acoustic, intense, dramatic"),
    ("Stick Season", "Noah Kahan", "folk", "nostalgic", 0.50, 118, 0.82, 0.66, 0.58, "Rustic New England acoustic folk anthem with driving acoustic guitar and autumn storytelling.", "folk, acoustic, guitar, autumn, mountain, hiking, nostalgic"),
    ("The Night We Met", "Lord Huron", "folk", "sorrowful", 0.37, 87, 0.10, 0.54, 0.97, "Haunting waltz-time indie folk ballad drenched in acoustic guitar reverb and midnight heartache.", "folk, acoustic, guitar, sad, sorrowful, melancholic, existential dread, late night, rain"),
    ("Skinny Love", "Bon Iver", "folk", "melancholic", 0.26, 76, 0.11, 0.59, 0.84, "Intimate cabin-recorded resonator acoustic guitar ballad with fragile falsetto and rainy introspection.", "folk, acoustic, guitar, melancholic, rain, rainy night, introspection, somber"),
    ("Holocene", "Bon Iver", "folk", "peaceful", 0.30, 148, 0.14, 0.38, 0.95, "Contemplative fingerpicked acoustic guitar and vibraphone landscape evoking mountain snow and quiet clarity.", "folk, acoustic, guitar, peaceful, study, coding, calm, meditation, mountain"),
    ("Fast Car", "Tracy Chapman", "folk", "nostalgic", 0.29, 104, 0.52, 0.71, 0.81, "Iconic fingerpicked acoustic guitar storytelling ballad about hope, escape, and midnight city drives.", "folk, acoustic, guitar, nostalgic, night drive, chill"),
    ("Here Comes The Sun", "The Beatles", "folk", "happy", 0.54, 129, 0.39, 0.56, 0.03, "Radiant acoustic-guitar classic celebrating spring sunshine, warmth, and renewal.", "folk, acoustic, guitar, happy, sunny, sunrise, morning"),
    ("Blackbird", "The Beatles", "folk", "peaceful", 0.13, 94, 0.37, 0.69, 0.75, "Delicate solo fingerstyle acoustic guitar piece accompanied by gentle birdsong foley.", "folk, acoustic, guitar, peaceful, calm, morning, study"),
    ("Yesterday", "The Beatles", "folk", "melancholic", 0.18, 97, 0.32, 0.33, 0.88, "Intimate acoustic guitar and string quartet ballad reflecting on longing and nostalgia.", "folk, classical, acoustic, strings, guitar, melancholic, nostalgic"),
    ("Landslide", "Fleetwood Mac", "folk", "nostalgic", 0.16, 159, 0.42, 0.41, 0.88, "Pure fingerpicked acoustic guitar reflection on time, mountain snow, and personal growth.", "folk, acoustic, guitar, nostalgic, peaceful, mountain"),
    ("Banana Pancakes", "Jack Johnson", "folk", "relaxed", 0.35, 114, 0.76, 0.80, 0.83, "Cozy rainy-morning acoustic guitar groove with real rain-on-roof foley and Sunday coffee warmth.", "folk, acoustic, guitar, rain, rainy, sunday, coffee, relaxed, chill"),

    # --- Lofi, Classical, Ambient & Deep Focus Study ---
    ("Clair de Lune", "Claude Debussy", "classical", "peaceful", 0.02, 66, 0.04, 0.30, 0.99, "Impressionistic solo grand piano nocturne evoking moonlight, stillness, and deep cognitive calm.", "classical, piano, acoustic, study, coding, sleep, meditation, peaceful, moonlight, night"),
    ("Nocturne Op. 9 No. 2", "Frédéric Chopin", "classical", "melancholic", 0.06, 68, 0.08, 0.34, 0.99, "Expressive solo piano nocturne with lyrical rubato melody ideal for late-night reading and introspection.", "classical, piano, acoustic, study, reading, library, night, melancholic, peaceful"),
    ("Gymnopédie No. 1", "Erik Satie", "classical", "chill", 0.01, 72, 0.09, 0.46, 0.99, "Minimalist slow solo piano piece providing weightless calm for pomodoro coding and anxiety relief.", "classical, lofi, piano, acoustic, study, coding, pomodoro, debugging, calm, sleep"),
    ("River Flows in You", "Yiruma", "classical", "romantic", 0.20, 68, 0.22, 0.38, 0.98, "Contemporary solo acoustic piano piece with flowing arpeggios and gentle rainy-night warmth.", "classical, piano, acoustic, study, coding, rain, romantic, peaceful"),
    ("Experience", "Ludovico Einaudi", "classical", "intense", 0.48, 92, 0.08, 0.25, 0.92, "Cinematic neoclassical piano and string orchestra crescendo building from quiet minimalism to dramatic emotional peak.", "classical, piano, strings, dramatic, melancholic, intense, study"),
    ("Nuvole Bianche", "Ludovico Einaudi", "classical", "peaceful", 0.10, 66, 0.12, 0.32, 0.99, "Serene solo felt-piano meditation crafted for deep focus, reading, and tranquil study sessions.", "classical, piano, acoustic, study, coding, focus, pomodoro, peaceful, calm"),
    ("Weightless", "Marconi Union", "ambient", "peaceful", 0.04, 60, 0.05, 0.15, 0.96, "Psychoacoustically engineered 60 BPM ambient piece using warm pads and harp harmonics to reduce anxiety and aid sleep.", "ambient, sleep, meditation, anxiety, calm, peaceful, stargazing, zero energy, study"),
    ("An Ending (Ascent)", "Brian Eno", "ambient", "peaceful", 0.06, 64, 0.10, 0.18, 0.95, "Weightless space-ambient synthesizer composition evoking stargazing and tranquil zero-gravity drift.", "ambient, sleep, meditation, stargazing, spacewalk, calm, peaceful"),
    ("Snowfall", "Øneheart & reidenshi", "ambient", "melancholic", 0.12, 70, 0.09, 0.28, 0.92, "Atmospheric late-night ambient and lofi soundscape for rainy coding, solitude, and quiet reflection.", "ambient, lofi, rain, rainy night, coding, study, melancholic, somber, late night"),
    ("Coffee Breath", "Sofia Mills", "lofi", "chill", 0.30, 78, 0.55, 0.72, 0.85, "Warm bedroom lofi-pop track with muted acoustic guitar, vinyl crackle, and cozy rainy cafe vibes.", "lofi, chill, coffee, cafe, rain, study, coding, acoustic, guitar"),
    ("Death Bed (Coffee for Your Head)", "Powfu & beabadoobee", "lofi", "chill", 0.43, 144, 0.68, 0.73, 0.80, "Lo-fi hip-hop study beat sampling an acoustic indie ballad with warm piano and rain ambiance.", "lofi, chill, coffee, study, coding, rain, acoustic, piano"),
    ("Luv(sic) Part 3", "Nujabes & Shing02", "lofi", "nostalgic", 0.58, 95, 0.74, 0.76, 0.68, "Foundational jazz-hop and lofi classic blending warm acoustic piano loops with soulful boom-bap drums.", "lofi, jazz, hip hop, study, coding, pomodoro, piano, nostalgic, chill"),
    ("Three Little Birds", "Bob Marley & The Wailers", "reggae", "happy", 0.51, 74, 0.88, 0.82, 0.09, "Uplifting roots-reggae classic with warm organ skank, sunny island rhythm, and stress-free positivity.", "reggae, beach, island, tropical, happy, relaxed, sunny"),
    ("Despacito / Bailando (English Mix)", "Enrique Iglesias & Sean Paul", "latin", "festive", 0.78, 100, 0.89, 0.80, 0.18, "Festive Latin-pop and acoustic Spanish guitar dancefloor anthem built for summer rooftop celebrations.", "latin, festive, party, dance, summer, guitar, happy"),
]


def build_top_100_english_catalog(start_id: int = 18) -> List[Dict[str, Any]]:
    """Return the 100 curated English hits formatted as Muse-ic song dictionaries with YTM links."""
    tracks: List[Dict[str, Any]] = []
    for offset, row in enumerate(TOP_100_ENGLISH_RAW):
        title, artist, genre, mood, energy, bpm, valence, dance, acoustic, desc, themes = row
        search_q = urllib.parse.quote(f"{title} {artist}")
        tracks.append(
            {
                "id": start_id + offset,
                "title": title,
                "artist": artist,
                "genre": genre,
                "mood": mood,
                "energy": float(energy),
                "tempo_bpm": float(bpm),
                "valence": float(valence),
                "danceability": float(dance),
                "acousticness": float(acoustic),
                "description": f"[Top 100 English] {desc}",
                "themes": f"{themes}, {artist.lower()}, {title.lower()}, top 100 english",
                "youtube_url": f"https://music.youtube.com/search?q={search_q}",
                "playlist": "Top 100 English",
            }
        )
    return tracks

