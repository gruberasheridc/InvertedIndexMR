import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.commons.validator.routines.UrlValidator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

public class InvertedIndex extends Configured implements Tool {

	private static final Logger LOG = Logger.getLogger(InvertedIndex.class);
	private static final String WORD_MARKER = "W"; // Used to indicate that a string is a word.
	private static final String URL_MARKER = "U"; // Used to indicate that a string is a URL.
	private static final String WORD_TYPE_SEPERATOR = "#";
	private static final String WORD_SKIP_PATTERNS = "word.skip.patterns"; // system variable specifying if a list of word patterns to skip is provided.
	private static final String SKIP_PATTERNS_FLAGE = "-skip"; // command line argument specifying if the skip patter functionality should be active. 

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new InvertedIndex(), args);
		System.exit(res);
	}

	public int run(String[] args) throws Exception {
		Job job = Job.getInstance(getConf(), "invertedindex");
		for (int i = 0; i < args.length; i += 1) {
			if (SKIP_PATTERNS_FLAGE.equals(args[i])) {
				// The skip word patterns functionality is active, load the list of word patterns from the specified path.
				LOG.info("Skip word patterns functionality is active.");
				job.getConfiguration().setBoolean(WORD_SKIP_PATTERNS, true);
				i += 1;
				job.addCacheFile(new Path(args[i]).toUri());
				LOG.info("Added file to the distributed cache: " + args[i]);
			}
		}

		job.setJarByClass(this.getClass());
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.setMapperClass(InvertedIndexMap.class);
		job.setReducerClass(InvertedIndexReduce.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.getConfiguration().set("mapreduce.output.textoutputformat.separator", ","); // Change output format from TSV to CSV.

		return job.waitForCompletion(true) ? 0 : 1;
	}

	public static class InvertedIndexMap extends Mapper<LongWritable, Text, Text, Text> {
		private final static Text one = new Text("1");
		private boolean caseSensitive = false;
		private Set<String> patternsToSkip = new HashSet<String>();

		// Create a regular expression pattern used to parse each line of input text on word boundaries.
		private static final Pattern WORD_BOUNDARY = Pattern.compile(" ");

		@Override
		protected void setup(Mapper<LongWritable, Text, Text, Text>.Context context) throws IOException, InterruptedException {
			if (context.getInputSplit() instanceof FileSplit) {
				((FileSplit) context.getInputSplit()).getPath().toString();
			} else {
				context.getInputSplit().toString();
			}
			
			Configuration config = context.getConfiguration();
			this.caseSensitive = config.getBoolean("wordcount.case.sensitive", false); // Get the case sensitive system variable from the command line execution variable.
			if (config.getBoolean(WORD_SKIP_PATTERNS, false)) {
				// The system variable is true, get the list of patterns to skip and add them to the list.
				URI[] localPaths = context.getCacheFiles();
				parseSkipFile(localPaths[0]);
			}
		}

		private void parseSkipFile(URI patternsURI) {
			LOG.info("Added file to the distributed cache: " + patternsURI);
			
			BufferedReader fis = null;
			try {
				fis = new BufferedReader(new FileReader(new File(patternsURI.getPath()).getName()));
				String pattern;
				while ((pattern = fis.readLine()) != null) {
					patternsToSkip.add(pattern);
				}
			} catch (IOException ioe) {
				LOG.error("Caught exception while parsing the cached file " + patternsURI + ".", ioe);
			} finally {
				if (fis != null) {
					try {
						fis.close();
					} catch (IOException e) {
						LOG.error("Failed to close pattern file. ", e);
					}
				}
			}
		}

		@Override
		public void map(LongWritable offset, Text lineText, Context context) throws IOException, InterruptedException {
			String line = lineText.toString();
			if (!caseSensitive) {
				line = line.toLowerCase();
			}

			// Get the owner site URL (the site the word was taken from) that is always the first word of the line.
			int firstWordIdx = line.indexOf(" ");
			String ownerSite = line.substring(0, firstWordIdx);
			Text site = new Text(ownerSite);
			line = line.replaceFirst(ownerSite, "");
			
			// Go over the words and process according to type.
			String [] words = WORD_BOUNDARY.split(line);			
			for (int i = 1; i < words.length; i++) {
				String word = words[i]; // Get the current word to process.
				if (word.isEmpty() || patternsToSkip.contains(word)) {
					continue;
				}

				Text currentWord = null;
				UrlValidator urlValidator = new UrlValidator();
				if (urlValidator.isValid(word)) {
					// The word is a URL. Mark it as one and count its occurrence.
					currentWord = new Text(word + WORD_TYPE_SEPERATOR + URL_MARKER);
					context.write(currentWord, one);
				} else {
					// We have an actual word. Mark it as one and collect the current site.
					currentWord = new Text(word + WORD_TYPE_SEPERATOR + WORD_MARKER);
					context.write(currentWord, site);
				}
			}
		}
	}

	public static class InvertedIndexReduce extends Reducer<Text, Text, Text, Text> {

		@Override
		public void reduce(Text word, Iterable<Text> occurences, Context context) throws IOException, InterruptedException {
			String [] wordParts = word.toString().split(WORD_TYPE_SEPERATOR);
			Text result = null;			
			String wordType = wordParts[1];
			switch (wordType) {
			case URL_MARKER:
				// When the word is a URL we should collect the number of occurrences the URL was referenced.
				result = collectURLOccurences(occurences);
				break;
			case WORD_MARKER:
				// When the word is regular, we should collect all the sites the word was taken from (owner sites).
				result = collectWordSites(occurences);
				break;
			};

			Text currWord = new Text(wordParts[0]);
			context.write(currWord, result);
		}

		private Text collectURLOccurences(Iterable<Text> occurences) {
			Integer numOfOccurences = 0;
			for (Text count : occurences) {
				int num = Integer.valueOf(count.toString());
				numOfOccurences += num;
			}

			Text urlOccurences = new Text(numOfOccurences.toString());
			return urlOccurences;
		}

		private Text collectWordSites(Iterable<Text> occurences) {
			Set<String> sites = new HashSet<String>();
			for (Text site : occurences) {
				sites.add(site.toString());
			}

			Text joindSites = new Text(join(sites, ","));
			return joindSites;
		}

		private static <T> String join(final Collection<T> objs, final String delimiter) {
			if (objs == null || objs.isEmpty())
				return "";

			Iterator<T> iter = objs.iterator();
			if (!iter.hasNext())
				return "";

			StringBuffer buffer = new StringBuffer(String.valueOf(iter.next()));
			while (iter.hasNext())
				buffer.append(delimiter).append(String.valueOf(iter.next()));

			return buffer.toString();
		}
	}
}