/**
 * The nineteen Hindi books the Recurrent page's Hindi networks learn from, all on Hindi
 * Wikisource (https://hi.wikisource.org). Every one was first published before 1931 and
 * its author or translator died long before 1966, so it is in the public domain in India
 * (life plus sixty years) and in the United States (published before 1931), like the
 * English books. Folk and children's tales come first, then stories and novels.
 *
 * `page` is the Wikisource title; `year` the first publication (approximate for the
 * oldest), not the year of the scanned edition; `skip` names chapters that are a later
 * editor's introduction rather than the book; `scan` reads a range of scan pages directly,
 * for a book whose chapters are not all set up as pages.
 */
export const HINDI_BOOKS = [
  { slug: 'tilismati-mundari', page: 'तिलिस्माती मुँदरी', title: 'Tilismati Mundari (The Magic Ring)', author: 'Shridhar Pathak', year: 1927 },
  { slug: 'rani-ketki', page: 'रानी केतकी की कहानी', title: 'Rani Ketki ki Kahani', author: 'Insha Allah Khan', year: 1803 },
  { slug: 'raja-bhoj-ka-sapna', page: 'राजा भोज का स्वप्ना', title: 'Raja Bhoj ka Sapna', author: 'Raja Shivprasad ‘Sitara-e-Hind’', year: 1856 },
  { slug: 'prem-sagar', page: 'प्रेमसागर', title: 'Prem Sagar', author: 'Lallu Lal', year: 1810, skip: ['जीवन-चरित', 'गद्य साहित्य का विकास'] },
  { slug: 'saptasaroj', page: 'सप्तसरोज', title: 'Saptasaroj (stories)', author: 'Premchand', year: 1917 },
  { slug: 'panch-phool', page: 'पाँच फूल', title: 'Panch Phool (stories)', author: 'Premchand', year: 1929 },
  { slug: 'prem-panchami', page: 'प्रेम पंचमी', title: 'Prem Panchami (stories)', author: 'Premchand', year: 1930 },
  { slug: 'nirmala', page: 'निर्मला', title: 'Nirmala', author: 'Premchand', year: 1927 },
  { slug: 'premashram', page: 'प्रेमाश्रम', title: 'Premashram', author: 'Premchand', year: 1922 },
  { slug: 'kayakalp', page: 'कायाकल्प', title: 'Kayakalp', author: 'Premchand', year: 1926 },
  { slug: 'akashdeep', page: 'आकाश-दीप', title: 'Akashdeep (stories)', author: 'Jaishankar Prasad', year: 1929 },
  { slug: 'chandrakanta-santati-2', page: 'चन्द्रकांता सन्तति २', title: 'Chandrakanta Santati, volume 2', author: 'Devaki Nandan Khatri', year: 1896 },
  { slug: 'chandrakanta-santati-3', page: 'चन्द्रकांता सन्तति 3', title: 'Chandrakanta Santati, volume 3', author: 'Devaki Nandan Khatri', year: 1896 },
  { slug: 'chandrakanta-santati-4', page: 'चन्द्रकांता सन्तति 4', title: 'Chandrakanta Santati, volume 4', author: 'Devaki Nandan Khatri', year: 1896 },
  { slug: 'chandrakanta-santati-5', page: 'चन्द्रकान्ता सन्तति 5', title: 'Chandrakanta Santati, volume 5', author: 'Devaki Nandan Khatri', year: 1896 },
  // Its scan is proofread throughout, but only its first chapters have pages of their own.
  { slug: 'pariksha-guru', page: 'परीक्षा गुरु', title: 'Pariksha Guru', author: 'Lala Shrinivas Das', year: 1882, scan: { index: 'परीक्षा गुरु.djvu', from: 9, to: 320 } },
  { slug: 'sau-ajaan-ek-sujaan', page: 'सौ अजान और एक सुजान', title: 'Sau Ajaan aur Ek Sujaan', author: 'Balkrishna Bhatt', year: 1892 },
  { slug: 'kapalkundala', page: 'कपालकुण्डला', title: 'Kapalkundala (Hindi translation)', author: 'Bankim Chandra Chattopadhyay', year: 1900 },
  { slug: 'durgeshnandini', page: 'दुर्गेशनन्दिनी प्रथम भाग', title: 'Durgeshnandini, part 1 (Hindi translation)', author: 'Bankim Chandra Chattopadhyay, tr. Gadadhar Singh', year: 1882 },
];
